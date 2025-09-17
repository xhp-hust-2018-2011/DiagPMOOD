# setting the current working folder
import os, sys
# from IPython import get_ipython
# __file__ = get_ipython().user_ns['__file__']
TMP_DIR = os.path.dirname(os.path.abspath(__file__))
# TMP_DIR=r'D:\BaiduYunDownload\Openoodv1.5\xhp_filesv3'
sys.path.append(TMP_DIR)
ROOT_DIR=os.path.dirname(TMP_DIR)
sys.path.append(ROOT_DIR)
# package and predefined functions

# load the network
from openood.networks import ResNet50, Swin_T, ViT_B_16, RegNet_Y_16GF
from torchvision.models import ResNet50_Weights, Swin_T_Weights, ViT_B_16_Weights, RegNet_Y_16GF_Weights
from torch.hub import load_state_dict_from_url
import torch
import numpy as np
import argparse
import time

# load the functions from OOD functions, this is for ood score and other things
from ood_functions_v2 import get_fc
from ood_functions_v2 import compute_output,get_predcls,process_logits_cls
from ood_functions_v2 import acc,auc_and_fpr_recall
from ood_functions_v2 import EnergyScore#compute_ebo
from ood_functions_v2 import TemperatureScaling
from ood_functions_v2 import MaxLogitScore,MaxSoftmaxScore#compute_mlt,compute_msp
from ood_functions_v2 import ASHScore
from ood_functions_v2 import KNNPostprocessor
from ood_functions_v2 import process_ood_scores,process_ood_metrics_openood
from ood_functions_v2 import process_ood_metrics_openood_str,process_ood_metrics_openood_str2
from ood_functions_v2 import LeverageImportNeuron
from ood_functions_v2 import split_dict,concat_dict
from ood_functions_v2 import FSEnergyScore
# added
from ood_functions_v2 import KNNPostprocessor,NNGuidePostprocessor
from ood_functions_v2 import MDSPostprocessor,RMDSPostprocessor
from ood_functions_v2 import GENPostprocessor
from ood_functions_v2 import VIMPostprocessor
from ood_io import save_h5py,load_h5py
from ood_io import load_features
from ood_io import load_features_str
from ood_io import write_txt
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

def parse_args():
    parser = argparse.ArgumentParser(description='OOD Eavluation from features.')
    parser.add_argument('--root_dir',default=r'/mnt/d/BaiduYunDownload/Openoodv1.5',type=str)
    parser.add_argument('--feat_root',default=None,type=str)
    parser.add_argument('--feat_name', default='tvs2_reg-simpleaug1',type=str,
                        help='this is the name for the specific features')
    parser.add_argument('--id_name', type=str, default='imagenet200',choices=['cifar10','cifar100','imagenet','imagenet200'],
                        help='the path to the file to process')
    parser.add_argument('--arch',
                        default='resnet50',
                        choices=['resnet50', 'swin-t', 'vit-b-16', 'regnet-swav',
                                'mobilenetv2','regnet','convnext_s','wideresnet50',
                                'resnetxt50','shufflenetv2_x1_5',
                                'resnet18_32x32','resnet18_224x224','resnet18'],
                        help='this is only useful for imagenet')
    parser.add_argument('--tvs-version', default=1,type=str,
                        help='this is only useful for imagenet')#, choices=[1, 2]
    parser.add_argument('--tvs-pretrained', action='store_true',default=False)
    parser.add_argument('--model_path', type=str, 
                        default=r'/mnt/d/BaiduYunDownload/Openoodv1.5/xhp_results_old/imagenet_weaknet_resnet50_weakce_e30_lr0.01_train-fc-bs512/s99/best.ckpt',
                        help='the path to load trained models, if not use tvs pretrained')    
    parser.add_argument('--model_type', type=str, 
                        default='net',choices=['fc','net'],
                        help='the type of loaded models')   
    # parser.add_argument('--fc_from', type=str, choices=['fc','fcnet'],
    #                     default='fc',
    #                     help='where the fc layer comes from, both of tvs pretrained or not')   
    parser.add_argument('--result_root',default=None,type=str)
    args = parser.parse_args()
    return args

# this only test one model
def oodtest_main(args):
    id_name=args.id_name
    arch=args.arch
    # skip_train=True # since just test, no need to load the train features
    skip_train=False # since just test, no need to load the train features

    # ----------------------------------------
    # load the feature
    # ----------------------------------------
    if args.feat_root is None:
        args.feat_root = args.root_dir
    FEAT_ROOT = os.path.join(args.feat_root)
    feat_name = args.feat_name
    # if args.id_name == 'imagenet':
    #     feat_name = os.path.join(f"{args.id_name}-{args.arch}-{args.feat_name}")
    #     args.feat_name = feat_name
    # else:
    #     feat_name = os.path.join(f"{args.id_name}-{args.feat_name}")
    #     args.feat_name = feat_name
    
    # ========================================
    # here should extract feature if not exist
    # ========================================
    # added20250514
    feat_abspath = os.path.join(FEAT_ROOT,feat_name)
    # if not os.path.exists(feat_abspath):
    #     if f'{args.id_name}-' in feat_name:
    #         feat_name = feat_name.replace(f'{args.id_name}-',f'{args.id_name}_',)
    #     elif f'{args.id_name}_' in args.model_path:
    #         feat_name = feat_name.replace(f'{args.id_name}_',f'{args.id_name}-',)
    # feat_abspath = os.path.join(FEAT_ROOT,feat_name)
    # if not os.path.exists(feat_abspath):
    #     if feat_name[-3:-1]=='_s':
    #         feat_name = feat_name[:-3]+feat_name[-1]
    #     else:
    #         feat_name = feat_name[:-1]+'_s'+feat_name[-1]
    # feat_abspath = os.path.join(FEAT_ROOT,feat_name)
    # if not os.path.exists(feat_abspath):
    #     if f'{args.id_name}-' in feat_name:
    #         feat_name = feat_name.replace(f'{args.id_name}-',f'{args.id_name}_',)
    #     elif f'{args.id_name}_' in args.model_path:
    #         feat_name = feat_name.replace(f'{args.id_name}_',f'{args.id_name}-',)
    feat_dict = load_features_str(FEAT_ROOT,
                            feat_dir = feat_name,
                              skip_train=skip_train,
                              )
    for tkey in list(feat_dict.keys()):
        if len(feat_dict[tkey]['feat_list'].shape)==4:
            feat_dict[tkey]['feat_list']=feat_dict[tkey]['feat_list'].squeeze(-1).squeeze(-1)
    
    
    # ----------------------------------------
    # load the feature (actually only fc layer is required)
    # ----------------------------------------    
    if args.id_name == 'imagenet':
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
            preprocessor = weights.transforms()   
        elif args.arch == 'convnext_s':
            net = convnext_s() 
            weights = eval(
                f'ConvNeXt_Small_Weights.IMAGENET1K_V{args.tvs_version}')
            net.load_state_dict(load_state_dict_from_url(weights.url))
            preprocessor = weights.transforms()       
        elif args.arch == 'wideresnet50':
            net = Wide_ResNet50()
            weights = eval(f'Wide_ResNet50_2_Weights.IMAGENET1K_V{args.tvs_version}')
            net.load_state_dict(load_state_dict_from_url(weights.url))
            preprocessor = weights.transforms()        
        elif args.arch == 'resnetxt50':
            net = ResNetXt50()
            weights = eval(f'ResNeXt50_32X4D_Weights.IMAGENET1K_V{args.tvs_version}')
            net.load_state_dict(load_state_dict_from_url(weights.url))
            preprocessor = weights.transforms() 
        elif args.arch == 'shufflenetv2_x1_5':
            net = shufflenetv2_x1_5()
            weights = eval(f'ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V{args.tvs_version}')
            net.load_state_dict(load_state_dict_from_url(weights.url))
            preprocessor = weights.transforms() 
        else:
            raise NotImplementedError
        loaded_model = None
        if args.tvs_pretrained and args.model_type == 'net': # tvs_pretrained weights
            pass           
        else: # must load weights, either fc or whole network
            MODEL_PATH = args.model_path
            ckpt = torch.load(MODEL_PATH)
            if args.model_type == 'net':
                try:
                    ckpt = ckpt['model'] # this is the torchvision save
                except:
                    ckpt = ckpt # this is for openood model
                net.load_state_dict(ckpt)  
            elif args.model_type == 'fc':
                ckpt = ckpt['fcnet']
                fc_layer = get_fc(net)
                fc_layer.weight.data = ckpt['weight'].data
                fc_layer.bias.data = ckpt['bias'].data
          
        # if args.fc_from == 'fcnet':
        #     if loaded_model is None:
        #         MODEL_PATH = args.model_path
        #         loaded_model = torch.load(MODEL_PATH)                
        #     fclayer = get_fc(net)
        #     fclayer.weight.data = loaded_model['fcnet.weight'].cpu()
        #     fclayer.bias.data = loaded_model['fcnet.bias'].cpu()
        # elif args.fc_from == 'fc':
        #     pass
    else: 
        if args.id_name == 'imagenet200':
            net = ResNet18_224x224(num_classes=200)  
        elif  args.id_name == 'cifar100':
            net = ResNet18_32x32(num_classes=100)              
        elif  args.id_name == 'cifar10':
            net = ResNet18_32x32(num_classes=10) 
        else:
            print("No such dataset")
        if not (args.model_path is None):
            # added20250514
            # if not os.path.exists(args.model_path):
            #     if f'{args.id_name}-' in args.model_path:
            #         args.model_path=args.model_path.replace(f'{args.id_name}-',f'{args.id_name}_',)
            #     elif f'{args.id_name}_' in args.model_path:
            #         args.model_path=args.model_path.replace(f'{args.id_name}_',f'{args.id_name}-',)
            # if not os.path.exists(args.model_path):
            #     # if args.model_path[-3:-1]=='_s':
            #     #     args.model_path = args.model_path[:-3]+args.model_path[-1]
            #     # else:
            #     #     args.model_path = args.model_path[:-1]+'_s'+args.model_path[-1]
            #     if args.model_path[-len('best_fc.pth')-4:-len('best_fc.pth')-2]=='_s':
            #         args.model_path = args.model_path[:-len('best_fc.pth')-4]+args.model_path[-len('best_fc.pth')-2:]
            #     else:
            #         args.model_path = args.model_path[:-len('best_fc.pth')-2]+'_s'+args.model_path[-len('best_fc.pth')-2:]
            # if not os.path.exists(args.model_path):
            #     if f'{args.id_name}-' in args.model_path:
            #         args.model_path=args.model_path.replace(f'{args.id_name}-',f'{args.id_name}_',)
            #     elif f'{args.id_name}_' in args.model_path:
            #         args.model_path=args.model_path.replace(f'{args.id_name}_',f'{args.id_name}-',)
            ckpt = torch.load(args.model_path, map_location='cpu')
            if args.model_type == 'net':
                try:
                    ckpt = ckpt['model'] # this is the torchvision save
                except:
                    ckpt = ckpt # this is for openood model
                net.load_state_dict(ckpt)  
            elif args.model_type == 'fc':
                ckpt = ckpt['fcnet']
                fc_layer = get_fc(net)
                fc_layer.weight.data = ckpt['weight'].data
                fc_layer.bias.data = ckpt['bias'].data
    # --------------------------
    #compute the output based on feratures
    # --------------------------
    feat_dict = process_logits_cls(feat_dict,net,MAX_SIZE=1e4,skip_train=skip_train)
    
    # ---------------------------------------------------------------------
    # load the existing result
    # ---------------------------------------------------------------------
    if args.result_root is None:
        args.result_root = args.root_dir
    if not os.path.exists(args.result_root):
        os.makedirs(args.result_root)

    result_name = feat_name
    if args.model_type == 'net':
        result_save_path = os.path.join(args.result_root,f'{result_name}-net.pth')
    elif args.model_type == 'fc':
        result_save_path = os.path.join(args.result_root,f'{result_name}-fc.pth')
    
    if os.path.exists(result_save_path):
        save_dict = torch.load(result_save_path)
    else:
        save_dict = dict()

    # ---------------------------------------------------------------------
    # Initilaize the ood score functions (search for the best parameters)
    # ---------------------------------------------------------------------
    # 1.first initialize some functions also set up
    TempScale = TemperatureScaling()
    TempScale.setup(feat_dict['id-val'])

    

    ebot = 1
    temperature_list = [0.01,0.1,1,5,10,100,1000]
    percentile_list = list(range(5,100,5))#[5,10,15,20,25,30,35,40,45,50,55,60,65, 70, 75, 80, 85, 90, 95]
    # param_dict={'temperature':ebot}

    

    # param_dict= {**param_dict,**{'clip_th':0.8,'pa':10,'pw':10}}


    param_dict={'temperature':ebot}
    search_dict = {'temperature':temperature_list}
    ebo = EnergyScore(param_dict = param_dict,search_dict = search_dict)
    # ebo.hyperparam_search(feat_dict=feat_dict)

    mlt = MaxLogitScore()

    param_dict={'temperature':ebot}
    search_dict = {'temperature':temperature_list}
    msp = MaxSoftmaxScore(param_dict = param_dict,search_dict = search_dict)
    # msp.hyperparam_search(feat_dict=feat_dict)


    threshold = np.quantile(feat_dict['id-val']['feat_list'], 0.92)
    param_dict={'clip_th':1}
    search_dict={'clip_th':[1,threshold,1e10]}
    fsebo = FSEnergyScore(net,param_dict=param_dict,search_dict = search_dict)   
    fsebo.hyperparam_search(feat_dict=feat_dict)

    # search_dict = {}
    # search_dict = {'K':[50, 100, 200, 500, 1000]}
    # knn = KNNPostprocessor(param_dict={'K':50,'alpha': 1.0},search_dict=search_dict)
    # knn.setup(feat_dict['id-train']['feat_list'])
    # knn.hyperparam_search(feat_dict=feat_dict)

    # search_dict = {}
    # search_dict = {'K':[50, 100, 200, 500, 1000]}
    # knn50 = KNNPostprocessor(param_dict={'K':50,'alpha': 1.0},search_dict=search_dict)
    # knn50.setup(feat_dict['id-train']['feat_list'])


    nnguide = NNGuidePostprocessor(param_dict={ 'K': 100,'alpha': 0.01 })
    nnguide.setup(feat_dict['id-train']['feat_list'],feat_dict['id-train']['logit_list'])

    # param_dict = {'gamma':0.1,'M':100}
    # search_dict = { 'gamma':[0.01,0.1,0.5,1,2,5,10],'M':[10,50,100,200,500,1000]} 
    # gen = GENPostprocessor(param_dict=param_dict,search_dict = search_dict)
    # gen.hyperparam_search(feat_dict=feat_dict)


    # param_dict = {'dim':256}
    # search_dict = { 'dim':[256,1000]} 
    # vim = VIMPostprocessor(param_dict=param_dict,search_dict = search_dict)
    # vim.setup(net,feat_dict['id-train']['feat_list'],feat_dict['id-train']['logit_list'])    
    # vim.hyperparam_search(feat_dict=feat_dict)


    # 2.register the ood forward functions, need to unify the input
    score_func_dict = {
        'mltscore':mlt,#compute_mlt,
        'eboscore':ebo,#compute_ebo,
        'mspscore':msp,
        'tempscore':TempScale,#TempScale.postprocess,
        'fseboscore':fsebo,
        # 'knnscore':knn,
        # 'knn50score':knn50,
        'nnguidescore':nnguide,
        # 'genscore':gen,
        # 'vimscore':vim,
    }



    # ---------------------------------------------------------------------
    # compute the ood score
    # ---------------------------------------------------------------------
    for score_name in list(score_func_dict.keys()):
        # skip the saved score
        if ('id-test' in save_dict) and (score_name in save_dict['id-test']):
            print('score has already been processed in save_dict')
            continue
        start_time = time.time()
        score_func = score_func_dict[score_name]
        # feat_dict = process_ood_scores(feat_dict,score_name,score_func.postprocess,param_dict=score_func.param_dict,\
        #                                MAX_SIZE=500,skip_train=skip_train)
        feat_dict = process_ood_scores(feat_dict,score_name,score_func.postprocess,param_dict=score_func.param_dict,\
                                       MAX_SIZE=500,skip_train=True)
        end_time = time.time()
        duration = end_time - start_time
        print(f"{score_name} takes {duration:.2f}s")
    
    
    # save the GT, logits and ood scores
    print('Now saving the ood results')     
    
    for skey in list(feat_dict.keys()):
        if skey == 'id-train':
            continue
        save_dict[skey] =  dict()
        for ikey in list(feat_dict[skey]):
            if  ('score' in ikey) \
                or (ikey in ['label_list','imname_list','logit_list']):
                save_dict[skey][ikey] = feat_dict[skey][ikey]
    # print(save_dict)
    if not os.path.exists(result_save_path):
        torch.save(save_dict,result_save_path)
    else:
        print(f"{result_save_path} already exists")
        torch.save(save_dict,result_save_path) ## added now, just incremental add

    # ---------------------------------------------------------------------
    # compute the ood metric, print and write
    # ---------------------------------------------------------------------
    if args.model_path is None:
        ood_save_path = os.path.join(ROOT_DIR,'Features','ood_%s_tvsv%d.txt' %(arch,args.tvs_version) )
        fsood_save_path = os.path.join(ROOT_DIR,'Features','fsood_%s_tvsv%d.txt' %(arch,args.tvs_version) )
    else:
        dirname = os.path.dirname(args.model_path)
        ood_save_path = os.path.join(dirname,'ood_%s.txt' %(arch) )
        fsood_save_path = os.path.join(dirname,'fsood_%s.txt' %(arch) )
    print_str = process_ood_metrics_openood_str(feat_dict,fsood=False)
    write_str = write_txt(print_str,ood_save_path,mode='w')
    # add one which save things in a single file
    if not os.path.exists(args.total_txtpath):
        mode = 'w'
    else:
        mode = 'a'
    print_str,ret_dict = process_ood_metrics_openood_str2(feat_dict,fsood=False)
    print_str = f'{args.feat_name}\n{args.model_path}\n'+print_str+'=='*20+'\n'
    write_str = write_txt(print_str,args.total_txtpath,mode=mode)        

    if 'imagenet' in args.id_name:
        # print_str = process_ood_metrics_openood(feat_dict,fsood=True) 
        print_str = process_ood_metrics_openood_str(feat_dict,fsood=True)   
        write_str = write_txt(print_str,fsood_save_path,mode='w')
    return ret_dict 


# this test multiple seeds of one model, and give the avg (all) results
# receive a list and then return the mean (overall results)
def oodtest_main_group(args,feat_name_list=None,model_path_list=None):
    tmp_idname = args.id_name
    arch=args.arch
    model_type=args.model_type
    # create the txt_path to write the results
    tmp_folder = args.result_root # save in the result_root
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    if tmp_idname == 'imagenet':
        args.total_txtpath = os.path.join(tmp_folder,f'{tmp_idname}-{model_type}-ood_results.txt')
    else:
        args.total_txtpath = os.path.join(tmp_folder,f'{tmp_idname}-{arch}-{model_type}-ood_results.txt')
    
    # if os.path.exists(args.total_txtpath):
    #     write_txt('',args.total_txtpath,mode='a')
    # else:
    #     write_txt('',args.total_txtpath,mode='w')    

    if os.path.exists(args.total_txtpath.replace('.txt','_mean.txt')):
        write_txt('',args.total_txtpath.replace('.txt','_mean.txt'),mode='a')
    else:
        write_txt('',args.total_txtpath.replace('.txt','_mean.txt'),mode='w')    

    # first loop the model
    for pidx in range(len(feat_name_list)):
        path = feat_name_list[pidx][0]
        res_dict = {}
        res_dict[path] = {}
        for midx in range(len(feat_name_list[pidx])):
            model_path = model_path_list[pidx][midx]
            args.model_path=model_path     
            args.feat_name = feat_name_list[pidx][midx]                       
            tmp_dict = oodtest_main(args)
            # initilaize 
            if len(res_dict[path])<1:
                for tkey in list(tmp_dict.keys()):
                    res_dict[path][tkey] =  []
            # loop for different ood scores
            for tkey in list(res_dict[path].keys()):
                res_dict[path][tkey].append(tmp_dict[tkey])
        # now save the avg results
        print('avg results')
        tmp_str = '=='*20+'\n'
        tmp_str += f'{path}\n'
        tmp_str += '--'*20+'\n'
        for tkey in list(res_dict[path].keys()):
            res_dict[path][tkey] = np.array(res_dict[path][tkey])
            tmp_str += '%-10s\t%5s\t%5s\t%5s\t  %5s\t%5s\t%5s' %(tkey,'N-AUC','F-AUC','ID Acc','N-FPR','F-FPR','ID Acc') +'\n'
            for ri in range(res_dict[path][tkey].shape[0]):
                mean = res_dict[path][tkey][ri]
                tmp_str += f"{tkey+f'-{ri}':10s}\t{mean[0]:5.2f}\t{mean[1]:5.2f}\t{mean[2]:5.2f}\t  {mean[3]:5.2f}\t{mean[4]:5.2f}\t{mean[5]:5.2f}" +'\n'                
            
            mean = np.mean(res_dict[path][tkey],axis=0)
            std = np.std(res_dict[path][tkey],axis=0)
            tmp_str += '%-10s\t%5s\t%5s\t%5s\t  %5s\t%5s\t%5s' %(tkey,'N-AUC','F-AUC','ID Acc','N-FPR','F-FPR','ID Acc') +'\n'
            tmp_str += f"{tkey+'-mean':10s}\t{mean[0]:5.2f}\t{mean[1]:5.2f}\t{mean[2]:5.2f}\t  {mean[3]:5.2f}\t{mean[4]:5.2f}\t{mean[5]:5.2f}" +'\n'
            tmp_str += f"{tkey+'-std':10s}\t{std[0]:5.2f}\t{std[1]:5.2f}\t{std[2]:5.2f}\t  {std[3]:5.2f}\t{std[4]:5.2f}\t{std[5]:5.2f}" +'\n'
            tmp_str += r"%-10s\t%5.2f$\pm$%5.2f\t%5.2f$\pm$%5.2f\t%5.2f$\pm$%5.2f\t%5.2f$\pm$%5.2f\t%5.2f$\pm$%5.2f\t%5.2f$\pm$%5.2f" \
                %(tkey,mean[0],std[0],mean[1],std[1],mean[2],std[2],mean[3],std[3],mean[4],std[4],mean[5],std[5])+'\n'
            print(tmp_str)
        tmp_save_path = args.total_txtpath.replace('.txt','_mean.txt')
        if not os.path.exists(tmp_save_path):
            mode = 'w'
        else:
            mode = 'a'
        write_str = write_txt(tmp_str,tmp_save_path,mode=mode)

if __name__=='__main__':
    args = parse_args()
    # args.root_dir = r'D:\BaiduYunDownload\OODwork202505\augrev_simplified20240915'
    args.root_dir = r'/mnt/d/BaiduYunDownload/OODwork202505/augrev_simplified20240915'

    list1 = ['imagenet200']
    list2 = ['net']
    args.arch='resnet18'
    for id_name,model_type in zip(list1[:],list2[:]):
        args.id_name = id_name
        args.model_type = model_type             
        args.feat_root = os.path.join("./",'Features') # then will add '/Features/'
        args.result_root = os.path.join("./",f'ScoreResults-{args.id_name}')
        name_list = [
            "imagenet200_v1_s1",
            "imagenet200_ls_s1",
            "imagenet200_mixup_s1",
            "imagenet200_mixup_ls_s1",
            "imagenet200_v2_s1",
            "imagenet200_augrevise_s1",
            ]
        feat_name_list = []
        model_path_list = []

        for name in name_list:
            feat_name_list.append([])
            model_path_list.append([])
            model_root = os.path.join("./",f'models')
            if args.model_type == 'net':
                model_path=f"{model_root}/{name}/model_best.pth" 
                if name == "imagenet200_v1_s1":
                    model_path=f"{model_root}/{name}/best.ckpt" 
            else:
                model_path=f"{model_root}/wa-{name}/best_fc.pth" 
            model_path_list[-1].append(model_path)
            
            feat_name = f'{name}'
            feat_name_list[-1].append(feat_name)
        
        
        oodtest_main_group(args,feat_name_list=feat_name_list,model_path_list=model_path_list)

    list1 = ['imagenet']
    list2 = ['net']
    args.arch='resnet50'
    for id_name,model_type in zip(list1[:],list2[:]):
        args.id_name = id_name
        args.model_type = model_type             
        args.feat_root = os.path.join("./",'Features') # then will add '/Features/'
        args.result_root = os.path.join("./",f'ScoreResults-{args.id_name}')
        name_list = [
            "imagenet_resnet50_augrev_m100",
            ]
        feat_name_list = []
        model_path_list = []

        for name in name_list:
            feat_name_list.append([])
            model_path_list.append([])
            model_root = os.path.join("./",f'models')
            if args.model_type == 'net':
                model_path=f"{model_root}/{name}/model_best.pth" 
            else:
                model_path=f"{model_root}/wa-{name}/best_fc.pth" 
            model_path_list[-1].append(model_path)
            
            feat_name = f'{name}'
            feat_name_list[-1].append(feat_name)
        
        
        oodtest_main_group(args,feat_name_list=feat_name_list,model_path_list=model_path_list)

    list1 = ['imagenet']
    list2 = ['net']
    args.arch='swin-t'
    for id_name,model_type in zip(list1[:],list2[:]):
        args.id_name = id_name
        args.model_type = model_type             
        args.feat_root = os.path.join("./",'Features') # then will add '/Features/'
        args.result_root = os.path.join("./",f'ScoreResults-{args.id_name}')
        name_list = [
            "imagenet_swin-t_augrev_m10",
            ]
        feat_name_list = []
        model_path_list = []

        for name in name_list:
            feat_name_list.append([])
            model_path_list.append([])
            model_root = os.path.join("./",f'models')
            if args.model_type == 'net':
                model_path=f"{model_root}/{name}/model_best.pth" 
            else:
                model_path=f"{model_root}/wa-{name}/best_fc.pth" 
            model_path_list[-1].append(model_path)
            
            feat_name = f'{name}'
            feat_name_list[-1].append(feat_name)
        
        
        oodtest_main_group(args,feat_name_list=feat_name_list,model_path_list=model_path_list)