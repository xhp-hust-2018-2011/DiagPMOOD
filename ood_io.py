import h5py 
import numpy as np
import torch
import os
import glob

def write_txt(text_str,file_path,mode = 'w'):
    with open(file_path, mode, encoding='utf-8') as file:
        file.write(text_str)
    return


# ====================================================
# this is for save a plain dict
# ====================================================
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

# make it a simple equal depth dict
def load_features_str(FEAT_ROOT='./Features',feat_dir='cifar100-resnet50-base-s0',skip_train=False):
    # feat_dir = os.path.join(FEAT_ROOT,f'FeaturesTVS1-{arch}')
    feat_dir = os.path.join(FEAT_ROOT,feat_dir)
    feat_dict = {}
    
    for setname in ['id','csid','ood']:
        if not os.path.exists(os.path.join(feat_dir,setname)):
            continue
        subdirs = os.listdir(os.path.join(feat_dir,setname))
        for subdir in subdirs:
            if not ('.h5' in subdir):
                continue
            if setname == 'id' or setname == 'csid':
                # idname,_ = subdir.split('_')
                del_len = len('_features.h5')
                idname = subdir[:-del_len]
                feat_str = f"{setname}-{idname}"
                if setname=='id' and idname=='train' and skip_train:
                    continue  
                feat_path = os.path.join(feat_dir,setname,subdir)
            else:
                del_len = len('_features.h5')
                idname = subdir[:-del_len]
                if 'val_' in subdir:
                    del_len = len('val_')
                    idname = idname[del_len:]
                    feat_str = f"{setname}-val"
                    feat_path = os.path.join(feat_dir,setname,subdir)
                elif 'near_' in subdir:
                    del_len = len('near_')
                    idname = idname[del_len:]
                    feat_str = f"{setname}-near-{idname}"
                    feat_path = os.path.join(feat_dir,setname,subdir)
                elif 'far_' in subdir:
                    del_len = len('far_')
                    idname = idname[del_len:]
                    feat_str = f"{setname}-far-{idname}"
                    feat_path = os.path.join(feat_dir,setname,subdir)
                else:
                    pass    
            feat_dict[feat_str] = load_h5py(feat_path)                
    return feat_dict

# make it a simple equal depth dict
def load_features(FEAT_ROOT='./Features', arch='resnet50',skip_train=False,feat_name=None):
    if feat_name is None:
        feat_dir = os.path.join(FEAT_ROOT,f'FeaturesTVS1-{arch}')
    else:
        feat_dir = os.path.join(FEAT_ROOT,feat_name)
    
    feat_dict = {}
    # load id
    setname = 'id'
    for idname in ['train','val','test']:
        if idname=='train' and skip_train:
            continue
        feat_str = f"{setname}-{idname}"
        feat_path = os.path.join(feat_dir,setname,f"{idname}_features.h5")
        feat_dict[feat_str] = load_h5py(feat_path)
    
    # load csid
    setname = 'csid'
    for idname in ['imagenet_c','imagenet_r','imagenet_v2']:
        feat_str = f"{setname}-{idname}"
        feat_path = os.path.join(feat_dir,setname,f"{idname}_features.h5")
        feat_dict[feat_str] = load_h5py(feat_path)

    # load ood-val
    setname = 'ood'
    idname = 'val'
    feat_str = f"{setname}-{idname}"
    feat_path = os.path.join(feat_dir,setname,f"{idname}_features.h5")
    feat_dict[feat_str] = load_h5py(feat_path)

    # load ood-near
    subsetname = 'near'
    for idname in ['ninco','ssb_hard']:
        feat_str = f"{setname}-{subsetname}-{idname}"
        feat_path = os.path.join(feat_dir,setname,f"{subsetname}_{idname}_features.h5")
        feat_dict[feat_str] = load_h5py(feat_path)
    # load ood-near
    subsetname = 'far'
    for idname in ['inaturalist','openimage_o','textures']:
        feat_str = f"{setname}-{subsetname}-{idname}"
        feat_path = os.path.join(feat_dir,setname,f"{subsetname}_{idname}_features.h5")
        feat_dict[feat_str] = load_h5py(feat_path)
    return feat_dict

def save_features_key(feat_dict,feat_dir,keystr ='score'):
    feat_subdir = os.path.join(feat_dir,keystr)
    if not os.path.exists(feat_subdir):
        os.makedirs(feat_subdir)    
    for dsets in list(feat_dict.keys()):
        new_dict = {}
        new_dict[keystr] = feat_dict[dsets][keystr]
        feat_path = os.path.join(feat_subdir,f"{dsets}.h5")
        save_h5py(new_dict,feat_path)
    return

def load_feature_key(feat_dir,keystr ='score'):
    feat_dict = {}
    feat_subdir = os.path.join(feat_dir,keystr)
    # glob the file_list
    file_list = glob.glob(feat_subdir,'*.h5')
    for fpath in file_list:
        fname_with_extension = os.path.basename(fpath)
        fname = os.path.splitext(fname_with_extension)[0]
        feat_dict[fname] = load_h5py(fpath)
    return feat_dict

def decode_bstr(bstr):
    decode_str = bstr.decode('utf-8', errors='replace')
    return decode_str

