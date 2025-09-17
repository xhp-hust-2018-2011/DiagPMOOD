import collections
import os, sys
# ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# sys.path.append(ROOT_DIR)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
import numpy as np
import pandas as pd
import argparse
# import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet50_Weights, Swin_T_Weights, ViT_B_16_Weights, RegNet_Y_16GF_Weights
from torchvision import transforms as trn
from torch.hub import load_state_dict_from_url
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

from openood.evaluation_api import Evaluator

from openood.networks import ResNet50, Swin_T, ViT_B_16, RegNet_Y_16GF
from openood.networks.conf_branch_net import ConfBranchNet
from openood.networks.godin_net import GodinNet
from openood.networks.rot_net import RotNet
from openood.networks.cider_net import CIDERNet


from openood.evaluation_api.datasets import get_id_ood_dataloader
from openood.evaluators import get_evaluator
from dotmap import DotMap
import h5py 
import os
from tqdm import tqdm
# import the network
from openood.networks import ResNet18_32x32, ResNet18_224x224
from openood.evaluation_api.preprocessor import get_default_preprocessor

def save_h5py(in_data,save_path):
    # Save the dictionary to an HDF5 file
    with h5py.File(save_path, 'w') as h5file:
        for key, value in in_data.items():
            h5file.create_dataset(key, data=value)

def load_h5py(load_path):
    # Load the HDF5 file and read data into a new dictionary
    loaded_data = {}
    with h5py.File(load_path, 'r') as h5file:
        for key in h5file.keys():
            loaded_data[key] = h5file[key][:]   
        # change bstr to str use decode "strings.decode('utf-8')"
    return loaded_data

def to_np(x):
    return x.data.cpu().numpy()

def extract_feature_from_dataloader(
            net,
            data_loader,
            filename = 'feature'):
    net.eval()
    feat_list, label_list = [], []
    imagename_list = []

    dlen = len(data_loader)
    # with alive_bar(len(data_loader)) as bar:
    with torch.no_grad():
        for batch in tqdm(data_loader):
        # for idx,batch in enumerate((data_loader)):
            data = batch['data'].cuda()
            label = batch['label']
            _, feat = net(data, return_feature=True)
            feat_list.extend(to_np(feat))
            label_list.extend(to_np(label))
            imagename_list+=batch['image_name']
            # print(f"{idx+1}/{dlen}")

    feat_list = np.array(feat_list)
    label_list = np.array(label_list)

    return_dict = {'feat_list':feat_list,
            'label_list':label_list,
            'imname_list':imagename_list}
    return return_dict

# this is for updating the results of dicts or collections
def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


parser = argparse.ArgumentParser()
parser.add_argument('--id_name',default='imagenet',choices=['cifar10','cifar100','imagenet200','imagenet'])
parser.add_argument('--arch',
                    default='resnet50',
                    choices=['resnet50', 'swin-t', 'vit-b-16', 'regnet-swav',
                             'mobilenetv2','regnet','convnext_s','wideresnet50',
                             'resnetxt50','shufflenetv2_x1_5','resnet18'],
                    help='this is only useful for imagenet')
parser.add_argument('--tvs-version', default=1,type=str,
                    help='this is only useful for imagenet')#, choices=[1, 2]
parser.add_argument('--tvs-pretrained', action='store_true',default=False)
parser.add_argument('--ckpt-path',
                    default="/mnt/d/BaiduYunDownload/Openoodv1.5/torchvision_code/torchvision_code_20240720/resnet50_swint_tvs2_ftall_allaug/imagenet_resnet50_tvs2_reg-simpleaug_s1/model_best.pth", 
                    help='this is loading the checkpoints if not using tvs model')
parser.add_argument('--save-path', default='tvs2_reg-simpleaug1')
parser.add_argument('--postprocessor', default='msp')
# parser.add_argument('--save-csv', action='store_true')
# parser.add_argument('--save-score', action='store_true')
# parser.add_argument('--fsood', action='store_true')
parser.add_argument('--batch-size', default=200, type=int)
# parser.add_argument('--shuffle', action='store_true') # this will be False, since just extract feature
parser.add_argument('--num_workers',type=int,default=2) 
parser.add_argument('--skip_train', action='store_true',default=False)
args = parser.parse_args()



# ==================
# set the folder #
# ==================
'''
   ROOT_DIR: openood root
   root: for saving results
'''

# here the root is for saving results （this should be an input arg）
# if not args.tvs_pretrained:
#     assert args.ckpt_path is not None
#     root = '/'.join(args.ckpt_path.split('/')[:-1])
# else:
#     root = os.path.join(
#         ROOT_DIR, 'test_results',
#         f'imagenet_{args.arch}_tvsv{args.tvs_version}_base_default')
#     if not os.path.exists(root):
#         os.makedirs(root)

root = ROOT_DIR


feat_root = os.path.join(ROOT_DIR,"Features",f"{args.save_path}")
if not os.path.exists(feat_root):
    os.makedirs(feat_root)



# specify an implemented postprocessor
# 'openmax', 'msp', 'temp_scaling', 'odin'...
# postprocessor_name = args.postprocessor




# =========================================================
# set the model (net) and data transform (preprocessor) #
# =========================================================
# assuming the model is either
# 1) torchvision pre-trained; or
# 2) a specified checkpoint
if args.id_name == 'imagenet':
    # if args.tvs_pretrained:
    if args.arch == 'resnet50':
        net = ResNet50()
        weights = eval(f'ResNet50_Weights.IMAGENET1K_V{args.tvs_version}')
        net.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
    elif args.arch == 'swin-t':
        net = Swin_T()
        weights = eval(f'Swin_T_Weights.IMAGENET1K_V{args.tvs_version}')
        net.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
    elif args.arch == 'vit-b-16':
        net = ViT_B_16()
        weights = eval(f'ViT_B_16_Weights.IMAGENET1K_V{args.tvs_version}')
        net.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
    elif args.arch == 'regnet-swav':
        net = RegNet_Y_16GF()
        weights = eval(
            f'RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V{args.tvs_version}')
        net.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
    elif args.arch == 'regnet':
        net = RegNet_Y_16GF()
        weights = eval(
            f'RegNet_Y_16GF_Weights.IMAGENET1K_V{args.tvs_version}')
        net.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
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
    # else:
    if args.tvs_pretrained:
        pass
    else:
        # net = ResNet50()
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        try:
            ckpt = ckpt['model'] # this is the torchvision save
        except:
            ckpt = ckpt # this is for openood model
        net.load_state_dict(ckpt)
    preprocessor = trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
elif args.id_name in ['cifar10','cifar100','imagenet200']:
    NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'imagenet200': 200}
    MODEL = {
        'cifar10': ResNet18_32x32,
        'cifar100': ResNet18_32x32,
        'imagenet200': ResNet18_224x224,
    }
    num_classes = NUM_CLASSES[args.id_name]
    model_arch = MODEL[args.id_name]
    net = model_arch(num_classes=num_classes)
    preprocessor = get_default_preprocessor(args.id_name)
    if not (args.ckpt_path is None):
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        try:
            ckpt = ckpt['model'] # this is the torchvision save
        except:
            ckpt = ckpt # this is for openood model
        net.load_state_dict(ckpt)

net.cuda()
net.eval()


# =========================================================
# set the dataset #
# =========================================================
id_name= args.id_name
data_root=os.path.join(ROOT_DIR, 'data')
loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': False,#args.shuffle,
    'num_workers': args.num_workers,
}
dataloader_dict = get_id_ood_dataloader(id_name, data_root,
                                        preprocessor, **loader_kwargs)
# print(dataloader_dict)


# =========================================================
# testing the acc #
# =========================================================
# # config = configparser.ConfigParser()
# # config['evaluator.name'] = 'base'
# config =dict(evaluator=dict(name='base'),
#              output_dir=os.path.join(ROOT_DIR,'features'))
# # '/code/OOD/OOD_Datasets/Openood_data/features'

# config = DotMap(config)
# evaluator = get_evaluator(config)


# =========================================================
# extract the feature #
# =========================================================
# extract all the files (exclude imagenet train)
if not os.path.exists(feat_root):
    os.makedirs(feat_root)
# begin from saving ID
for dkey in ['id','csid','ood']:
    if len(dataloader_dict[dkey])<1:
        continue
    sub_feat_root=os.path.join(feat_root,dkey)
    if not os.path.exists(sub_feat_root):
        os.makedirs(sub_feat_root)
    for tkey in list(dataloader_dict[dkey]):
        tmp_save_path = os.path.join(sub_feat_root,f'{tkey}_features.h5')
        if os.path.exists(tmp_save_path):
            # if tkey != 'train':
            #     tmp_feature = load_h5py(tmp_save_path)
            if dkey=='id' and tkey == 'test':
                tmp_feature = load_h5py(tmp_save_path)
                feat_list = torch.from_numpy(tmp_feature['feat_list'])
                pred_list = net.get_fc_layer()(feat_list.cuda())
                pred_idx = pred_list.max(dim=1)[1]
                id_acc = (tmp_feature['label_list'] == pred_idx.cpu().numpy()).mean()
                print(id_acc)
                pass
            continue
        if args.skip_train and dkey=='id' and tkey == 'train':
            continue
        # if dkey=='id':
        #     if tkey == 'train':
        #         continue
        if dkey=='ood':
            if tkey!='val':
                continue
        print(dataloader_dict[dkey][tkey])
        tmp_feature = extract_feature_from_dataloader(net,
                    dataloader_dict[dkey][tkey],
                    f'{args.id_name}_{args.arch}_feature')
        # print(tmp_feature)
        save_h5py(tmp_feature,tmp_save_path)
        print(f'{dkey}/{tkey} has been saved in {sub_feat_root}')

        if dkey=='id' and tkey == 'test':
            tmp_feature = load_h5py(tmp_save_path)
            feat_list = torch.from_numpy(tmp_feature['feat_list'])
            pred_list = net.get_fc_layer()(feat_list.cuda())
            pred_idx = pred_list.max(dim=1)[1]
            id_acc = (tmp_feature['label_list'] == pred_idx.cpu().numpy()).mean()
            print(id_acc)
            pass

# 20240808 should add the idacc checking

# for ood, which is a second_dict
for dkey in ['ood']:#['id','csid','ood']:
    sub_feat_root=os.path.join(feat_root,dkey)
    if not os.path.exists(sub_feat_root):
        os.makedirs(sub_feat_root)
    for tkey in list(dataloader_dict[dkey]):
        if tkey == 'val':
            continue
        for ttkey in list(dataloader_dict[dkey][tkey]):
            tmp_save_path = os.path.join(sub_feat_root,f'{tkey}_{ttkey}_features.h5')
            if os.path.exists(tmp_save_path):
                continue
            tmp_feature = extract_feature_from_dataloader(net,
                        dataloader_dict[dkey][tkey][ttkey],
                        f'{args.id_name}_{args.arch}_feature')
            save_h5py(tmp_feature,tmp_save_path)
            print(f'{dkey}/{tkey}-{ttkey} has been saved in {sub_feat_root}')