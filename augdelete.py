# setting the current working folder
import os, sys
# from IPython import get_ipython
# __file__ = get_ipython().user_ns['__file__']
# TMP_DIR = os.path.dirname(os.path.abspath(__file__))
# TMP_DIR=r'D:\BaiduYunDownload\Openoodv1.5\xhp_filesv3'
# sys.path.append(TMP_DIR)
# ROOT_DIR=os.path.dirname(TMP_DIR)
# sys.path.append(ROOT_DIR)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
# package and predefined functions

# load the network
from openood.networks import ResNet50, Swin_T, ViT_B_16, RegNet_Y_16GF
from torchvision.models import ResNet50_Weights, Swin_T_Weights, ViT_B_16_Weights, RegNet_Y_16GF_Weights
from torch.hub import load_state_dict_from_url
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import time
from tqdm import tqdm
import json

# load the functions from OOD functions, this is for ood score and other things
from ood_functions_v2 import compute_output,get_predcls,process_logits_cls
from ood_functions_v2 import acc,auc_and_fpr_recall
from ood_functions_v2 import EnergyScore#compute_ebo
from ood_functions_v2 import TemperatureScaling
from ood_functions_v2 import MaxLogitScore,MaxSoftmaxScore#compute_mlt,compute_msp
from ood_functions_v2 import ASHScore
from ood_functions_v2 import KNNPostprocessor
from ood_functions_v2 import base_ood_eval
from ood_functions_v2 import process_ood_scores,process_ood_metrics_openood
from ood_functions_v2 import ash_s,ash_scale
from ood_functions_v2 import get_fc
from ood_functions_v2 import LeverageImportNeuron
from ood_functions_v2 import split_dict,concat_dict
from ood_io import save_h5py,load_h5py
from ood_io import load_features
from ood_io import write_txt
from feature_dataset import FeatDataset
# remember to set the random seed to pytorch and numpy  
# 20240711 added mobilenetv2
from torchvision.models import MobileNet_V2_Weights
from openood.networks.mobilenetv2 import mobilenetv2
from torchvision.models import ConvNeXt_Small_Weights
from openood.networks.convnext import convnext_s
from torchvision.models import Wide_ResNet50_2_Weights
from openood.networks.resnet50_others import Wide_ResNet50
from torchvision.models import ResNeXt50_32X4D_Weights
from openood.networks.resnet50_others import ResNetXt50
from torchvision.models import ShuffleNet_V2_X1_5_Weights
from openood.networks.shufflenetv2 import shufflenetv2_x1_5

from openood.networks import ResNet18_32x32, ResNet18_224x224
from ood_io import load_features_str
from ood_functions_v2 import process_ood_metrics_openood_str

class fc_layer(nn.Module):
    def __init__(self,in_dim,out_dim,layer_num=1):
        super().__init__()
        self.layer_num = layer_num
        if self.layer_num ==1:
            self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            )
        elif self.layer_num ==2:
            self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
            )

    def forward(self,x):
        out = self.fc(x)
        return out    
    

class weak_oodscore(base_ood_eval):
    def __init__(self,net=None,param_dict={'score_name':'eboscore'},search_dict={}):
        super().__init__(param_dict,search_dict)
        self.net = net
        self.fcnet = net.fcnet
        

    def postprocess(self,feat_dict,param_dict={'score_name':'eboscore'},if_gpu=True):
        out_dict = {}

        # do a single forwad pass
        tmp_feat = feat_dict['feat_list']
        if isinstance(tmp_feat,np.ndarray):
            tmp_feat = torch.from_numpy(tmp_feat)

        # tmp_feat = tmp_feat.clamp(min=-1.0,max=1.0)
        if if_gpu:
            self.fcnet = self.fcnet.cuda()
            logit_dict = self.fcnet(tmp_feat.cuda())
        else:
            logit_dict = self.fcnet(tmp_feat)
        
        score = logit_dict
        score = score.max(dim=1)[0]
        score = score.detach().cpu()
        out_dict['mlt'] = score
        
        score = torch.logsumexp(logit_dict,dim=1)
        score = score.detach().cpu()
        out_dict['ebo'] = score

        score = F.softmax(logit_dict,dim=1)
        score = score.max(dim=1)[0]
        score = score.detach().cpu()
        out_dict['msp'] = score
        return out_dict            

      


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * \
                (1 + np.cos(step / total_steps * np.pi))

def concat_metrics(metrics_list):
    new_metric_dict = {}
    # initilize new_metric_dict
    sample_metrics = metrics_list[0]
    for tkey in list(sample_metrics.keys()):
        new_metric_dict[tkey] = []
        for idx in range(len(metrics_list)):
            tmp_metrics = metrics_list[idx]
            if isinstance(tmp_metrics[tkey],list):
                new_metric_dict[tkey] += tmp_metrics[tkey]
            else:
                new_metric_dict[tkey].append(tmp_metrics[tkey])
    return new_metric_dict


class BaseFeatTrainer():
    def __init__(self,net,train_loader,param_dict,config=None):
        self.net = net
        self.train_loader = train_loader
        self.param_dict = param_dict
        self.metrics_log = dict() 
        self.config = config

        # Check which parameters are frozen
        for name, param in net.named_parameters():
            if param.requires_grad:
                print(f"{name} is {'' if param.requires_grad else 'not '}trainable")        

        trainable_params = filter(lambda p: p.requires_grad, self.net.parameters())
        self.optimizer = torch.optim.SGD(
            trainable_params,
            param_dict['lr'],
            momentum=param_dict['momentum'],
            weight_decay=param_dict['weight_decay'],
            nesterov=True,
        )

        total_steps = param_dict['num_epochs']*len(train_loader)
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,
                1e-6 / param_dict['lr'],
            ),
        )

    def train_epoch(self,epoch_idx):
        # self.net.train()
        self.net.eval()
        self.net.fcnet.train()
        self.net = self.net.cuda()
        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)            

        for train_step in tqdm(range(1,
                                    len(train_dataiter) + 1),
                            desc='Epoch {:03d}: '.format(epoch_idx),
                            position=0,
                            leave=True):
            batch = next(train_dataiter)
            data = batch['data'].cuda() # here the loaded thing is the feature
            target = batch['label'].cuda()

            # forward with pretrained model
            with torch.no_grad():
                fc_layer = get_fc(self.net)
                old_logits = fc_layer(data)
                # get the mask
                cnum = old_logits.size()[1]
                BASE = torch.eye(cnum).type(old_logits.dtype).cuda()
                gt_mask = BASE[target.squeeze()]   
                gt_logit = (old_logits*gt_mask).sum(dim=1,keepdim=True)
                old_mask = (old_logits>=gt_logit)#.type(old_logits.dtype)

            # forward with the new model
            # right_mask = (old_mask.sum(dim=1,keepdim=False)>1)
            right_mask = (old_mask.sum(dim=1,keepdim=False)<2)

            logits = self.net.fcnet(data)
            # logits = old_logits
            masked_logits_max = logits.masked_fill(~old_mask, float('-inf')).max(dim=1)[0]
            # masked_loss = torch.log(torch.exp((logits-masked_logits_max.unsqueeze(1))).sum(dim=1)+1e-6).mean()
            masked_loss = torch.log(torch.exp((logits[~right_mask]-masked_logits_max[~right_mask].unsqueeze(1))).sum(dim=1)+1e-6).mean()


            loss = F.cross_entropy(logits, target.squeeze(1))

            # - here the loss contains multiple things
            loss_avg_dict = {'masked_loss':0.0}
            loss_name = 'masked_loss'
            loss_avg_dict[loss_name] = loss_avg_dict[loss_name]*0.8+ masked_loss.cpu().item()*0.2

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss.cpu().data) * 0.2

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = loss_avg
        for tkey in list(loss_avg_dict.keys()):
            metrics['loss_'+tkey] = loss_avg_dict[tkey]

        if len(self.metrics_log)==0:
            self.metrics_log = concat_metrics([metrics])
        else:
            self.metrics_log = concat_metrics([self.metrics_log,metrics])

        return self.net, metrics                


class BaseFeatEvaluator():
    def __init__(self,val_loader,param_dict={}):
        self.val_loader = val_loader
        self.param_dict = param_dict
        self.metrics_log = dict() # this id store the log
    
    def eval_acc(self,net,epoch_idx):
        with torch.no_grad():
            net = net.cuda()
            # accum_num = 0.0
            data_size = len(self.val_loader.dataset)
            loss_avg_dict = {}
            acc_avg_dict = {}
            loss_avg = 0.0
            loss = 0.0
            for batch in self.val_loader:
                data = batch['data'].cuda()
                target = batch['label'].cuda()
                tmp_dlen = data.size()[0]
                # accum_num += tmp_dlen
                # forward
                logits_dict = net(data)
                pred_cls = logits_dict.max(dim=1)[1]
                gt_cls = target[:,-1]
                tmp_acc = (pred_cls==gt_cls).float().mean().cpu().item()
                tmp_loss = F.cross_entropy(logits_dict, gt_cls)
                loss = tmp_loss
                loss_avg += loss.cpu().item()*(tmp_dlen/data_size)
                logit_name = 'cls'
                if not (logit_name in list(acc_avg_dict.keys()) ):
                    acc_avg_dict[logit_name] = tmp_acc*(tmp_dlen/data_size)
                else:
                    acc_avg_dict[logit_name] += tmp_acc*(tmp_dlen/data_size) 
                if not (logit_name in list(loss_avg_dict.keys()) ):
                    loss_avg_dict[logit_name] = tmp_loss.cpu().item()*(tmp_dlen/data_size)
                else:
                    loss_avg_dict[logit_name] += tmp_loss.cpu().item()*(tmp_dlen/data_size) 
                 
        metrics = dict()
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = loss_avg
        # metrics['loss_dict'] = loss_avg_dict
        # metrics['acc_dict'] = acc_avg_dict
        for tkey in list(loss_avg_dict.keys()):
            metrics['loss_'+tkey] = loss_avg_dict[tkey]
        for tkey in list(acc_avg_dict.keys()):
            metrics['acc_'+tkey] = acc_avg_dict[tkey]
        finest_str = "cls" 
        metrics['finest_loss'] = loss_avg_dict[finest_str]
        metrics['finest_acc'] = acc_avg_dict[finest_str]

        if len(self.metrics_log)==0:
            self.metrics_log = concat_metrics([metrics])
        else:
            self.metrics_log = concat_metrics([self.metrics_log,metrics])

        return metrics


# compute the logits and predicted labels
def process_logits_cls_netgeneral(feat_dict,net,if_gpu=True,MAX_SIZE=5e4,skip_train=False):
    # try to compute the metric of the ood samples
    net = net.eval()
    for tkey in list(feat_dict.keys()):
        if tkey=='id-train' and skip_train:
            continue
        tmp_feat_dict = feat_dict[tkey]
        # split the feat_dict
        tmp_feat_dict_list = split_dict(tmp_feat_dict,MAXL=MAX_SIZE)
        def compute_output(net,features,if_gpu=True):
            if isinstance(features,np.ndarray):
                features = torch.from_numpy(features)
            if if_gpu:
                out_dict = net(features.cuda())
            else:
                out_dict = net(features.cuda())
            logits = out_dict
            predcls = logits.max(dim=1)[1]
            logits = logits.detach().cpu()
            predcls = predcls.detach().cpu()      
            return {'logits':logits,
                    'predcls':predcls}
        for splited_feat_dict in tmp_feat_dict_list:
            tmp_outdict = compute_output(net.cuda(),splited_feat_dict['feat_list'],if_gpu=if_gpu)
            tmp_logits = tmp_outdict['logits']
            tmp_predcls = tmp_outdict['predcls']
            # instert to the dict
            splited_feat_dict['logit_list'] = tmp_logits.detach().numpy()
            splited_feat_dict['predcls'] = tmp_predcls.detach().numpy()
        feat_dict[tkey] = concat_dict(tmp_feat_dict_list)
    return feat_dict



def main(args):
    # load the feature 
    id_name=args.idname
    arch=args.arch
    skip_train=False

    # ----------------------------------------
    # load the feature
    # ----------------------------------------
    FEAT_ROOT = os.path.join(ROOT_DIR,'Features')
    feat_dict = load_features_str(FEAT_ROOT=FEAT_ROOT,feat_dir=args.feat_name,skip_train=skip_train,)
    for tkey in list(feat_dict.keys()):
        if len(feat_dict[tkey]['feat_list'].shape)==4:
            feat_dict[tkey]['feat_list']=feat_dict[tkey]['feat_list'].squeeze(-1).squeeze(-1)

    # -----------------------------
    # intialize the feat dataloader
    # -----------------------------
    set_key = 'id-train'
    feats = feat_dict[set_key]['feat_list']
    labels = feat_dict[set_key]['label_list']
    # need to transform the data label (label may combine to be N*2)

    train_set = FeatDataset(feats,labels)
    train_loader = DataLoader(train_set,batch_size=args.bs,shuffle=True)        

    set_key = 'id-val'
    feats = feat_dict[set_key]['feat_list']
    labels = feat_dict[set_key]['label_list']
    val_set = FeatDataset(feats,labels)
    val_loader = DataLoader(val_set,batch_size=args.bs,shuffle=False)

    set_key = 'id-test'
    feats = feat_dict[set_key]['feat_list']
    labels = feat_dict[set_key]['label_list']
    test_set = FeatDataset(feats,labels)
    test_loader = DataLoader(test_set,batch_size=args.bs,shuffle=False)    


    # -------------------------------------
    # intialize the hierarchy classifiers and optimizers
    # -------------------------------------
    # load the weight of the fc
    if args.idname == 'imagenet':
        tvs_version=args.tvs_version
        if arch == 'resnet50':
            net = ResNet50()
            weights = eval(f'ResNet50_Weights.IMAGENET1K_V{tvs_version}')
            net.load_state_dict(load_state_dict_from_url(weights.url))
        elif arch == 'swin-t':
            net = Swin_T()
            weights = eval(f'Swin_T_Weights.IMAGENET1K_V{tvs_version}')
            net.load_state_dict(load_state_dict_from_url(weights.url))
        elif arch == 'vit-b-16':
            net = ViT_B_16()
            weights = eval(f'ViT_B_16_Weights.IMAGENET1K_V{tvs_version}')
            net.load_state_dict(load_state_dict_from_url(weights.url))
        elif arch == 'regnet-swav':
            net = RegNet_Y_16GF()
            weights = eval(
                f'RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V{tvs_version}')
            net.load_state_dict(load_state_dict_from_url(weights.url))
        elif args.arch == 'regnet':
            net = RegNet_Y_16GF()
            weights = eval(
                f'RegNet_Y_16GF_Weights.IMAGENET1K_V{args.tvs_version}')
            net.load_state_dict(load_state_dict_from_url(weights.url))
        elif args.arch == 'mobilenetv2':
            net = mobilenetv2()
            weights = eval(
                f'MobileNet_V2_Weights.IMAGENET1K_V{args.tvs_version}')
            net.load_state_dict(load_state_dict_from_url(weights.url))   
        elif args.arch == 'convnext_s':
            net = convnext_s() 
            weights = eval(
                f'ConvNeXt_Small_Weights.IMAGENET1K_V{args.tvs_version}')
            net.load_state_dict(load_state_dict_from_url(weights.url))      
        elif args.arch == 'wideresnet50':
            net = Wide_ResNet50()
            weights = eval(f'Wide_ResNet50_2_Weights.IMAGENET1K_V{args.tvs_version}')
            net.load_state_dict(load_state_dict_from_url(weights.url))      
        elif args.arch == 'resnetxt50':
            net = ResNetXt50()
            weights = eval(f'ResNeXt50_32X4D_Weights.IMAGENET1K_V{args.tvs_version}')
            net.load_state_dict(load_state_dict_from_url(weights.url))    
        elif args.arch == 'shufflenetv2_x1_5':
            net = shufflenetv2_x1_5()
            weights = eval(f'ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V{args.tvs_version}')
            net.load_state_dict(load_state_dict_from_url(weights.url))
            preprocessor = weights.transforms() 
        else:
            raise NotImplementedError
        if not (args.model_path is None):
            ckpt = torch.load(args.model_path, map_location='cpu')
            try:
                ckpt = ckpt['model'] # this is the torchvision save
            except:
                ckpt = ckpt # this is for openood model
            net.load_state_dict(ckpt) 
        # freeze the pretrained network
        for name, param in net.named_parameters():
            param.requires_grad = False
        fc_layer = get_fc(net)
        in_ch,out_ch = fc_layer.in_features,fc_layer.out_features
        # new_layer = fc_layer(in_dim=in_ch,out_dim=out_ch,layer_num=1)
        setattr(net,'fcnet',nn.Linear(in_ch,out_ch)) # append the nmew_layer to the existing models
        if args.fc_pre:
            # set the weight
            net.fcnet.weight.data = fc_layer.weight.data.clone()
            net.fcnet.bias.data = fc_layer.bias.data.clone()
    elif args.idname == 'imagenet200':
        net = ResNet18_224x224(num_classes=200)        
        if not (args.model_path is None):
            ckpt = torch.load(args.model_path, map_location='cpu')
            try:
                ckpt = ckpt['model'] # this is the torchvision save
            except:
                ckpt = ckpt # this is for openood model
            net.load_state_dict(ckpt)  
        for name, param in net.named_parameters():
            param.requires_grad = False
        fc_layer = get_fc(net)
        in_ch,out_ch = fc_layer.in_features,fc_layer.out_features
        setattr(net,'fcnet',nn.Linear(in_ch,out_ch))
        if args.fc_pre:
            # set the weight
            net.fcnet.weight.data = fc_layer.weight.data.clone()
            net.fcnet.bias.data = fc_layer.bias.data.clone()
    elif  args.idname == 'cifar100':
        net = ResNet18_32x32(num_classes=100)        
        if not (args.model_path is None):
            ckpt = torch.load(args.model_path, map_location='cpu')
            try:
                ckpt = ckpt['model'] # this is the torchvision save
            except:
                ckpt = ckpt # this is for openood model
            net.load_state_dict(ckpt)  
        for name, param in net.named_parameters():
            param.requires_grad = False
        fc_layer = get_fc(net)
        in_ch,out_ch = fc_layer.in_features,fc_layer.out_features
        setattr(net,'fcnet',nn.Linear(in_ch,out_ch))
        if args.fc_pre:
            # set the weight
            net.fcnet.weight.data = fc_layer.weight.data.clone()
            net.fcnet.bias.data = fc_layer.bias.data.clone()
    elif  args.idname == 'cifar10':
        net = ResNet18_32x32(num_classes=10)        
        if not (args.model_path is None):
            ckpt = torch.load(args.model_path, map_location='cpu')
            try:
                ckpt = ckpt['model'] # this is the torchvision save
            except:
                ckpt = ckpt # this is for openood model
            net.load_state_dict(ckpt) 
        for name, param in net.named_parameters():
            param.requires_grad = False
        fc_layer = get_fc(net)
        in_ch,out_ch = fc_layer.in_features,fc_layer.out_features
        setattr(net,'fcnet',nn.Linear(in_ch,out_ch))
        if args.fc_pre:
            # set the weight
            net.fcnet.weight.data = fc_layer.weight.data.clone()
            net.fcnet.bias.data = fc_layer.bias.data.clone()
    # -------------------------------------
    # do the training & validation
    # -------------------------------------
    param_dict = {
        'lr':args.lr,#0.01,
        'momentum':0.9,
        'weight_decay':0.0005,
        'num_epochs':args.epoch_num,#15,
        }
    trainer = BaseFeatTrainer(net,train_loader,param_dict,config=args)
    evaluator = BaseFeatEvaluator(val_loader)
    test_evaluator = BaseFeatEvaluator(test_loader)

    save_name = ''
    if args.sample_use == 'all':
        save_name += 'wa'
    elif args.sample_use == 'right':
        save_name += 'wr'
    RES_ROOT = args.res_folder if args.res_folder else ROOT_DIR
    # save_folder = os.path.join(RES_ROOT,'refc-'+args.feat_name,save_name)
    save_folder = os.path.join(RES_ROOT,f'{save_name}-'+args.feat_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)    
    # load the existing checkpoints
    save_path = os.path.join(save_folder,'tmp_fc.pth')
    if os.path.exists(save_path):
        # load the model
        # net.load_state_dict(torch.load(save_path)['net'])  
        params = torch.load(save_path)
        net.fcnet.load_state_dict(params['fcnet'])
        fc_layer = get_fc(net)
        fc_layer.load_state_dict(params['fc'])      
        # load the log
        log_save_path = os.path.join(save_folder,'metrics_log.pth')
        log_dict = torch.load(log_save_path)
        trainer.metrics_log = log_dict['train_log']
        evaluator.metrics_log = log_dict['val_log']
        start_idx = log_dict['epoch_idx']+1
        epoch_idx = log_dict['epoch_idx']
    else:
        start_idx = 0

    for epoch_idx in range(start_idx,trainer.param_dict['num_epochs']):
        net,train_metrics = trainer.train_epoch(epoch_idx)
        # do evaluation
        val_metrics = evaluator.eval_acc(net.fcnet,epoch_idx)
        print(val_metrics)
        print(f"tmp acc: {val_metrics['finest_acc']}")
        # may also test the ood metrics

        # save the tmp model
        fc_layer = get_fc(net)        
        save_path = os.path.join(save_folder,'tmp_fc.pth')
        torch.save({'fc':fc_layer.state_dict(),'fcnet':net.fcnet.state_dict(),
                    'metrics':val_metrics},save_path)
        # also save the tmp statistics
        log_save_path = os.path.join(save_folder,'metrics_log.pth')
        torch.save({'train_log':trainer.metrics_log,'val_log':evaluator.metrics_log,'epoch_idx':epoch_idx},
                   log_save_path)
        # save the best model
        if evaluator.metrics_log['finest_acc'][-1] == max(evaluator.metrics_log['finest_acc']):
            save_path = os.path.join(save_folder,'best_fc.pth')
            torch.save({'fc':fc_layer.state_dict(),'fcnet':net.fcnet.state_dict(),
                        'metrics':val_metrics},save_path)           
        best_acc = max(evaluator.metrics_log['finest_acc'])
        print(f"best acc: {best_acc}")
    # do testing
    # load the best model
    save_path = os.path.join(save_folder,'best_fc.pth')
    params = torch.load(save_path)
    net.fcnet.load_state_dict(params['fcnet'])
    fc_layer = get_fc(net)
    fc_layer.load_state_dict(params['fc'])
    # test the best epoch
    test_metrics = test_evaluator.eval_acc(net.fcnet,epoch_idx)
    print(test_metrics)
    print(f"test acc: {test_metrics['finest_acc']}")    

    # -------------------------------------
    # do the ood evaluation (can finish later)
    # -------------------------------------
    # compute the logits and predicted labels
    del feat_dict['id-train']
    process_logits_cls_netgeneral(feat_dict,net.fcnet,if_gpu=True,MAX_SIZE=5e4,skip_train=True)
    
    
    # compute the score use the funcs in the ood_functions_v2.py
    MAX_SIZE = 1000
    weak_oodf = weak_oodscore(net = net) # the ood funcs
    score_func_dict = {
        'weak_oodscore':weak_oodf,
    }
    for score_name in list(score_func_dict.keys()):
        start_time = time.time()
        score_func = score_func_dict[score_name]
        feat_dict = process_ood_scores(feat_dict,score_name,score_func.postprocess,param_dict=score_func.param_dict,\
                                       MAX_SIZE=500,skip_train=True)
        end_time = time.time()
        duration = end_time - start_time
        print(f"{score_name} takes {duration:.2f}s")

    print_str = process_ood_metrics_openood_str(feat_dict,fsood=False)
    save_path = os.path.join(save_folder,'hood-%s.txt' %(arch) )
    write_str = write_txt(print_str,save_path,mode='w')

    if args.idname == 'imagenet' or args.idname == 'imagenet200':
        print_str = process_ood_metrics_openood_str(feat_dict,fsood=True)
        save_path = os.path.join(save_folder,'hood-%s.txt' %(arch) )
        write_str = write_txt(print_str,save_path,mode='a')

    return

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='OOD Eavluation from features.')
    parser.add_argument('--idname', type=str, default='cifar100',choices=['imagenet','cifar100','cifar10','imagenet200'],
                        help='the path to the file to process')
    parser.add_argument('--arch',
                        default='regnet',
                        choices=['resnet50', 'swin-t', 'vit-b-16', 'regnet-swav',
                                'mobilenetv2','regnet','convnext_s','wideresnet50',
                                'resnetxt50','shufflenetv2_x1_5','resnet18'],
                        help='this is only useful for imagenet')
    parser.add_argument('--feat_name', type=str, default='cifar100-allnoema1',
                        help='the path to the file to process')
    parser.add_argument('--tvs_version', type=int, default=1,
                        help='the tvs version')
    parser.add_argument('--model_path',default=None,type=str)
    parser.add_argument('--res_folder',default=None,type=str)
    parser.add_argument('--fc_pre',default=False,action='store_true')
    parser.add_argument('--bs', type=int, default=512,
                        help='the batch size of training')
    parser.add_argument('--lr', type=int, default=1e-2,
                        help='the batch size of training')
    parser.add_argument('--epoch_num', type=int, default=15,
                        help='the batch size of training')
    parser.add_argument('--label_tran',default=True,action='store_true')

    parser.add_argument('--label_path',type=str,default=r'D:\BaiduYunDownload\Openoodv1.5\data\benchmark_imglist\imagenet\train_imagenet_mos.txt',help='the path of new labels txt')
    parser.add_argument('--seed', type=int, default=99999,
                    help='the seed for training')
    parser.add_argument('--sample_use',type=str,choices=['all'],default='all')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)    

    # args.arch='resnet18_224x224'
    # args.idname = 'imagenet200'
    # args.res_folder = './wa_models'
    # for path in [
    #     "imagenet200_v1_s1",
    #     "imagenet200_ls_s1",
    #     "imagenet200_mixup_s1",
    #     "imagenet200_mixup_ls_s1",
    #     "imagenet200_v2_s1",
    #     "imagenet200_augrevise_s1",
    # ]:
    #     args.feat_name = path
    #     if path == "imagenet200_v1_s1":
    #         args.model_path=f"models/{path}/best.ckpt"
    #     else:
    #         args.model_path=f"models/{path}/model_best.pth"
    #     main(args)



    args.idname = 'imagenet'
    args.res_folder = './wa_models'
    for path in [
        "imagenet_resnet50_augrev_m100",
        "imagenet_swin-t_augrev_m10",
    ]:
        if 'resnet50' in path:
            args.arch='resnet50'
        elif 'swin-t' in path:
            args.arch='swin-t'
        args.feat_name = path
        args.model_path=f"models/{path}/model_best.pth"
        main(args)
