import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import metrics # this is for compute the metrics
import faiss # this is for knn search
from scipy.special import logsumexp
import copy
import sklearn.covariance
from tqdm import tqdm
from numpy.linalg import norm, pinv
from sklearn.covariance import EmpiricalCovariance

def split_dict(ori_dict,MAXL=1e4):
    dict_list = []

    key_list = list(ori_dict.keys())
    ORI_SIZE = len(ori_dict[key_list[0]])
    if ORI_SIZE<=MAXL:
        return [ori_dict]
    elif ORI_SIZE>=MAXL:
        split_num = int(np.ceil(ORI_SIZE/MAXL))
        for si in range(split_num):
            index1,index2 = MAXL*si,MAXL*(si+1)
            index1,index2 = int(index1),int(index2)
            index2 = min(index2,ORI_SIZE)
            new_dict = {}
            for tkey in key_list:
                new_dict[tkey] = ori_dict[tkey][index1:index2]  
            dict_list.append(new_dict)  
    return dict_list
# concat_dict from a list of dict
def concat_dict(dict_list):
    ori_dict = {}
    key_list = list(dict_list[0].keys())

    if len(dict_list)==1:
        return dict_list[0]
    elif len(dict_list)>1:
        for tkey in key_list:
            tkey_values = []
            for dict_idx in range(len(dict_list)):
                tkey_values.append(dict_list[dict_idx][tkey])
            if isinstance(tkey_values[0],np.ndarray):
                tkey_values = np.concatenate(tkey_values,axis=0)
            elif isinstance(tkey_values[0],list):
                tkey_values = sum(tkey_values,list)
            elif isinstance(tkey_values[0],torch.Tensor):
                tkey_values = torch.cat(tkey_values,dim=0)
            ori_dict[tkey] = tkey_values
    return ori_dict


# ==============================================
# process the logits and cls
# ==============================================

def get_fc(net):
    net_name = net.__class__.__name__
    if net_name == 'ResNet50':
        layer = net.fc
    elif net_name == 'ViT_B_16':
        layer = net.heads.head
    elif net_name == 'Swin_T':
        layer = net.head
    elif net_name == 'RegNet_Y_16GF':
        layer = net.fc
    elif 'ResNet' in net_name:
        layer = net.fc
    elif 'mobilenetv2' in net_name:
        layer = net.classifier[1]
    else:
        layer = net.get_fc_layer()
        # print("f{net_name} is not defined.")

    return layer




# now begin some simple computation
def compute_output(features,net,if_gpu=True):
    if isinstance(features,np.ndarray):
        features = torch.from_numpy(features)
    # features,net=features.cuda(),net.cuda()
    # net_name = net.__class__.__name__
    # if net_name == 'resnet':
    #     layer = net.fc()
    # elif net_name == 'VIT_B_16':
    #     layer = net.heads.head
    with torch.no_grad():
        layer = get_fc(net)
        if if_gpu:
            layer = layer.cuda()
            if len(features.size())==4:
                features=features.squeeze(-1).squeeze(-1)
            logits = layer(features.cuda())
            layer = layer.to('cpu')
            logits = logits.cpu()
        else:
            logits = layer(features)
    return logits

def get_pos_neg_logits(features,net,if_gpu=True):
    if isinstance(features,np.ndarray):
        features = torch.from_numpy(features) # N*2048
    layer = get_fc(net)
    weight,bias = layer.weight,layer.bias
    neg_mask = (weight<0).float()# 1000*2048
    pos_mask = 1- neg_mask
    if if_gpu:
        # layer = layer.cuda()
        neg_logits = torch.mm(features.cuda(),(neg_mask*weight).cuda().t())+bias.cuda()
        pos_logits = torch.mm(features.cuda(),(pos_mask*weight).cuda().t())+bias.cuda()
        neg_logits = neg_logits.cpu()
        pos_logits = pos_logits.cpu()
    else:
        neg_logits = torch.mm(features,(neg_mask*weight).t())+bias
        pos_logits = torch.mm(features,(pos_mask*weight).t())+bias
    return pos_logits, neg_logits



def get_predcls(logits,dim=1,if_gpu=True):
    if isinstance(logits,np.ndarray):
        logits = torch.from_numpy(logits)
    if if_gpu:
        predcls = logits.cuda().max(dim=dim)[1].cpu() # this return the index of the class
    else:
        predcls = logits.max(dim=dim)[1]
    return predcls


def process_logits_cls(feat_dict,net,if_gpu=True,MAX_SIZE=5e4,skip_train=False):
    # update 2024/04/24, need to define a maximum split size, here also do the slpit
    def get_output_predcls_onedict(splited_feat_dict,net,if_gpu=if_gpu):
        tmp_logits = compute_output(splited_feat_dict['feat_list'],net,if_gpu=if_gpu)
        tmp_predcls = get_predcls(tmp_logits,dim=1,if_gpu=if_gpu)
        # instert to the dict
        splited_feat_dict['logit_list'] = tmp_logits.detach().numpy()
        splited_feat_dict['predcls'] = tmp_predcls.detach().numpy()
        return

    # try to compute the metric of the ood samples
    net = net.eval()
    for tkey in list(feat_dict.keys()):
        if tkey=='id-train' and skip_train:
            continue
        tmp_feat_dict = feat_dict[tkey]
        # split the feat_dict
        tmp_feat_dict_list = split_dict(tmp_feat_dict,MAXL=MAX_SIZE)
        for splited_feat_dict in tmp_feat_dict_list:
            get_output_predcls_onedict(splited_feat_dict,net)
            # idx = np.argmax(splited_feat_dict['logit_list'],axis=1)==splited_feat_dict['label_list']
            # print(splited_feat_dict['logit_list'][idx].max(axis=1).mean())
            # print(splited_feat_dict['logit_list'][~idx].max(axis=1).mean())
        feat_dict[tkey] = concat_dict(tmp_feat_dict_list)
    return feat_dict



# ==============================================
# process the metrics
# ==============================================
# accuracy
def acc(pred, label):
    if isinstance(pred,torch.Tensor):
        pred = pred.numpy()
    if isinstance(label,torch.Tensor):
        pred = pred.numpy()
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc


# compute the AUROC, AUPR_IN, AUPR_OUT,FPR95
def auc_and_fpr_recall(conf, label, tpr_th=0.95):
    if isinstance(conf,torch.Tensor):
        conf = conf.numpy()
    if isinstance(label,torch.Tensor):
        conf = label.numpy()
    # following convention in ML we treat OOD as positive
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1 # OOD is 1 and ID is 0

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]  
    #In case of multiple occurrences of the maximum values, 
    #the indices corresponding to the first occurrence are returned.

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(ood_indicator, -conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(1 - ood_indicator, conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr




def process_ood_metrics_openood(feat_dict,fsood=False,IFSHOW=True,ret_res = False):
    CSID_NAME_LIST = ['imagenet_c','imagenet_r','imagenet_v2']
    OOD_FAR_LIST = ['inaturalist','openimage_o','textures']
    OOD_NEAR_LIST = ['ninco','ssb_hard']
    ret_dict = {}
    # feat dict should contain the score
    print_str = ''
    print_str += '\n' + "#"+"=="*20
    print_str += '\n' + "#"+"==" + "fsood begin"  if fsood else  '\n' + "#"+"==" + "general ood begin"
    print_str += '\n' + "#"+"=="*20
    # print_str += '%10s\t%5s\t%5s\t%5s' %('','auroc','fpr95','ID Acc')
    for tmp_score_name in list(feat_dict['id-test'].keys()):
        if not ('score' in tmp_score_name):
            continue 
        else:
            # construct printing str
            print_str += '\n' + "=="*20
            print_str += '\n' +  '%10s\t%5s\t%5s\t%5s' %(tmp_score_name,'auroc','fpr95','ID Acc')
            print_str += '\n' + "--"*20

            # compute the acc
            tkey='id-test'
            tmp_label = feat_dict[tkey]['label_list']
            tmp_pred = feat_dict[tkey]['predcls']
            if fsood:
                csid_label = []
                csid_pred = []
                # combine the csid to id
                for csid_name in CSID_NAME_LIST:
                    csid_key = f"csid-{csid_name}"
                    csid_label.append(feat_dict[csid_key]['label_list'])
                    csid_pred.append(feat_dict[csid_key]['predcls'])
                csid_label = np.concatenate(csid_label,axis=0)
                csid_pred = np.concatenate(csid_pred,axis=0)
                tmp_label = np.concatenate((tmp_label,csid_label),axis=0)
                tmp_pred = np.concatenate((tmp_pred,csid_pred),axis=0)
    

            test_acc = acc(tmp_pred, tmp_label)
            ret_dict['test_acc'] = [test_acc]
            # get the id right idx
            id_right_mask = (tmp_pred == tmp_label)
            # print(f'ID-{setkey}:{test_acc:.4f}')

            ood_metrics = {
                'far': [],
                'near': [],
                'far_r': [],
                'near_r': [],
                'far_f': [],
                'near_f': [],                
                'far_f+o': [],
                'near_f+o': [], 
                'idr_idf':[],
            }





            # setkey='test'
            set1 = feat_dict[tkey]
            tmp_score1 = set1[tmp_score_name]
            tmp_label1 = set1['label_list']
            if fsood:
                csid_label = []
                csid_score = []
                # combine the csid to id
                for csid_name in CSID_NAME_LIST:
                    csid_key = f"csid-{csid_name}"
                    csid_label.append(feat_dict[csid_key]['label_list'])
                    csid_score.append(feat_dict[csid_key][tmp_score_name])
                csid_label = np.concatenate(csid_label,axis=0)
                csid_score = np.concatenate(csid_score,axis=0)
                tmp_label1 = np.concatenate((tmp_label1,csid_label),axis=0)
                tmp_score1 = np.concatenate((tmp_score1,csid_score),axis=0)




            for name1 in ['far','near']:
                if name1=='far':
                    name2_list = OOD_FAR_LIST
                elif name1=='near':
                    name2_list = OOD_NEAR_LIST
                for name2 in name2_list:
                    set2_key = f"ood-{name1}-{name2}"
                    set2 = feat_dict[set2_key]
                    # --this is the id_all and ood
                    tmp_score2 = set2[tmp_score_name]
                    tmp_label2 = -1 * np.ones_like(set2['label_list'])
                    tmp_score = np.concatenate([tmp_score1,tmp_score2])
                    tmp_label = np.concatenate([tmp_label1,tmp_label2])
                    tmp_metric = auc_and_fpr_recall(tmp_score,tmp_label)
                    #print(name1,name2,np.array(tmp_metric))
                    ood_metrics[name1].append(tmp_metric)
                    auroc,fpr=tmp_metric[0],tmp_metric[-1]
                    # print_str += '\n' +  f"{name2:10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                    # -- this is the id-false and ood
                    tmp_score11 = tmp_score1[id_right_mask]
                    tmp_label11 = tmp_label1[id_right_mask]
                    tmp_score = np.concatenate([tmp_score11,tmp_score2])
                    tmp_label = np.concatenate([tmp_label11,tmp_label2])
                    tmp_metric = auc_and_fpr_recall(tmp_score,tmp_label)                    
                    ood_metrics[name1+'_r'].append(tmp_metric)
                    auroc,fpr=tmp_metric[0],tmp_metric[-1]
                    # print_str += '\n' +  f"{name2+'_r':10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"

                    tmp_score11 = tmp_score1[~id_right_mask]
                    tmp_label11 = tmp_label1[~id_right_mask]
                    tmp_score = np.concatenate([tmp_score11,tmp_score2])
                    tmp_label = np.concatenate([tmp_label11,tmp_label2])
                    tmp_metric = auc_and_fpr_recall(tmp_score,tmp_label)
                    ood_metrics[name1+'_f'].append(tmp_metric)
                    auroc,fpr=tmp_metric[0],tmp_metric[-1]
                    # print_str += '\n' +  f"{name2+'_f:10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"

                    tmp_score11 = tmp_score1[id_right_mask]
                    tmp_label11 = tmp_label1[id_right_mask]
                    tmp_score12 = tmp_score1[~id_right_mask]
                    tmp_label12 = 0*tmp_label1[~id_right_mask]-1
                    tmp_score = np.concatenate([tmp_score11,tmp_score12,tmp_score2])
                    tmp_label = np.concatenate([tmp_label11,tmp_label12,tmp_label2])
                    tmp_metric = auc_and_fpr_recall(tmp_score,tmp_label)
                    ood_metrics[name1+'_f+o'].append(tmp_metric)
                    auroc,fpr=tmp_metric[0],tmp_metric[-1]
                    # print_str += '\n' +  f"{name2+'_f+o:10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"


                ood_metrics[name1] = np.array(ood_metrics[name1])
                mean_metric = ood_metrics[name1].mean(axis=0)
                auroc,fpr=mean_metric[0],mean_metric[-1]
                print_str += '\n' +  f"{name1:10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                print_str += '\n' + "--"*20
                tmp_key = f"{tmp_score_name}-{name1}-auroc"
                ret_dict[tmp_key] = [auroc]
                tmp_key = f"{tmp_score_name}-{name1}-fpr"
                ret_dict[tmp_key] = [fpr]

                ood_metrics[name1+'_r'] = np.array(ood_metrics[name1+'_r'])
                mean_metric = ood_metrics[name1+'_r'].mean(axis=0)
                auroc,fpr=mean_metric[0],mean_metric[-1]
                print_str += '\n' +  f"{name1+'-idr':10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                print_str += '\n' + "--"*20

                ood_metrics[name1+'_f'] = np.array(ood_metrics[name1+'_f'])
                mean_metric = ood_metrics[name1+'_f'].mean(axis=0)
                auroc,fpr=mean_metric[0],mean_metric[-1]
                print_str += '\n' +  f"{name1+'-idf':10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                print_str += '\n' + "--"*20

                ood_metrics[name1+'_f+o'] = np.array(ood_metrics[name1+'_f+o'])
                mean_metric = ood_metrics[name1+'_f+o'].mean(axis=0)
                auroc,fpr=mean_metric[0],mean_metric[-1]
                print_str += '\n' +  f"{name1+'+idf-idr':10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                print_str += '\n' + "--"*20

                # idr-idf
                tmp_score11 = tmp_score1[id_right_mask]
                tmp_label11 = tmp_label1[id_right_mask]
                tmp_score12 = tmp_score1[~id_right_mask]
                tmp_label12 = 0*tmp_label1[~id_right_mask]-1
                tmp_score = np.concatenate([tmp_score11,tmp_score12])
                tmp_label = np.concatenate([tmp_label11,tmp_label12])
                tmp_metric = auc_and_fpr_recall(tmp_score,tmp_label)
                ood_metrics['idr_idf'].append(tmp_metric)
                auroc,fpr=tmp_metric[0],tmp_metric[-1]
                print_str += '\n' +  f"{'idr_idf':10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                print_str += '\n' + "--"*20

                # np.set_printoptions(precision=4)
    print_str += '\n' + "#"+"=="*20
    print_str += '\n' + "#"+"==" + "fsood end"  if fsood else  '\n' + "#"+"==" + "general ood end"
    print_str += '\n' + "#"+"=="*20
    if IFSHOW:
        print(print_str)
    
    if ret_res:
        return print_str,ret_dict
    else:
        return print_str#feat_dict

def process_ood_metrics_openood_str(feat_dict,fsood=False,IFSHOW=True):
    CSID_NAME_LIST = ['imagenet_c','imagenet_r','imagenet_v2']
    OOD_FAR_LIST = ['inaturalist','openimage_o','textures']
    OOD_NEAR_LIST = ['ninco','ssb_hard']
    # feat dict should contain the score
    print_str = ''
    print_str += '\n' + "#"+"=="*20
    print_str += '\n' + "#"+"==" + "fsood begin"  if fsood else  '\n' + "#"+"==" + "general ood begin"
    print_str += '\n' + "#"+"=="*20
    # print_str += '%10s\t%5s\t%5s\t%5s' %('','auroc','fpr95','ID Acc')
    for tmp_score_name in list(feat_dict['id-test'].keys()):
        if not ('score' in tmp_score_name):
            continue 
        else:
            # construct printing str
            print_str += '\n' + "=="*20
            print_str += '\n' +  '%10s\t%5s\t%5s\t%5s' %(tmp_score_name,'auroc','fpr95','ID Acc')
            print_str += '\n' + "--"*20

            # compute the acc
            tkey='id-test'
            tmp_label = feat_dict[tkey]['label_list']
            tmp_pred = feat_dict[tkey]['predcls']
            if fsood:
                csid_label = []
                csid_pred = []
                # combine the csid to id
                for csid_name in CSID_NAME_LIST:
                    csid_key = f"csid-{csid_name}"
                    csid_label.append(feat_dict[csid_key]['label_list'])
                    csid_pred.append(feat_dict[csid_key]['predcls'])
                csid_label = np.concatenate(csid_label,axis=0)
                csid_pred = np.concatenate(csid_pred,axis=0)
                tmp_label = np.concatenate((tmp_label,csid_label),axis=0)
                tmp_pred = np.concatenate((tmp_pred,csid_pred),axis=0)
    

            test_acc = acc(tmp_pred, tmp_label)
            # get the id right idx
            id_right_mask = (tmp_pred == tmp_label)
            # print(f'ID-{setkey}:{test_acc:.4f}')

            ood_metrics = {
                'far': [],
                'near': [],
                'far_r': [],
                'near_r': [],
                'far_f': [],
                'near_f': [],                
                'far_f+o': [],
                'near_f+o': [], 
                'idr_idf':[],
            }





            # setkey='test'
            set1 = feat_dict[tkey]
            tmp_score1 = set1[tmp_score_name]
            tmp_label1 = set1['label_list']
            if fsood:
                csid_label = []
                csid_score = []
                # combine the csid to id
                for csid_name in CSID_NAME_LIST:
                    csid_key = f"csid-{csid_name}"
                    csid_label.append(feat_dict[csid_key]['label_list'])
                    csid_score.append(feat_dict[csid_key][tmp_score_name])
                csid_label = np.concatenate(csid_label,axis=0)
                csid_score = np.concatenate(csid_score,axis=0)
                tmp_label1 = np.concatenate((tmp_label1,csid_label),axis=0)
                tmp_score1 = np.concatenate((tmp_score1,csid_score),axis=0)


            for name1 in ['far','near']:
                for tmp_key in list(feat_dict.keys()):
                    if not ('-'+name1+'-' in tmp_key):
                        continue
                    set2_key = tmp_key
                    set2 = feat_dict[set2_key]
                    # --this is the id_all and ood
                    tmp_score2 = set2[tmp_score_name]
                    tmp_label2 = -1 * np.ones_like(set2['label_list'])
                    tmp_score = np.concatenate([tmp_score1,tmp_score2])
                    tmp_label = np.concatenate([tmp_label1,tmp_label2])
                    tmp_metric = auc_and_fpr_recall(tmp_score,tmp_label)
                    #print(name1,name2,np.array(tmp_metric))
                    ood_metrics[name1].append(tmp_metric)
                    auroc,fpr=tmp_metric[0],tmp_metric[-1]
                    # print_str += '\n' +  f"{name2:10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                    # -- this is the id-false and ood
                    tmp_score11 = tmp_score1[id_right_mask]
                    tmp_label11 = tmp_label1[id_right_mask]
                    tmp_score = np.concatenate([tmp_score11,tmp_score2])
                    tmp_label = np.concatenate([tmp_label11,tmp_label2])
                    tmp_metric = auc_and_fpr_recall(tmp_score,tmp_label)                    
                    ood_metrics[name1+'_r'].append(tmp_metric)
                    auroc,fpr=tmp_metric[0],tmp_metric[-1]
                    # print_str += '\n' +  f"{name2+'_r':10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"

                    tmp_score11 = tmp_score1[~id_right_mask]
                    tmp_label11 = tmp_label1[~id_right_mask]
                    tmp_score = np.concatenate([tmp_score11,tmp_score2])
                    tmp_label = np.concatenate([tmp_label11,tmp_label2])
                    tmp_metric = auc_and_fpr_recall(tmp_score,tmp_label)
                    ood_metrics[name1+'_f'].append(tmp_metric)
                    auroc,fpr=tmp_metric[0],tmp_metric[-1]
                    # print_str += '\n' +  f"{name2+'_f:10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"

                    tmp_score11 = tmp_score1[id_right_mask]
                    tmp_label11 = tmp_label1[id_right_mask]
                    tmp_score12 = tmp_score1[~id_right_mask]
                    tmp_label12 = 0*tmp_label1[~id_right_mask]-1
                    tmp_score = np.concatenate([tmp_score11,tmp_score12,tmp_score2])
                    tmp_label = np.concatenate([tmp_label11,tmp_label12,tmp_label2])
                    tmp_metric = auc_and_fpr_recall(tmp_score,tmp_label)
                    ood_metrics[name1+'_f+o'].append(tmp_metric)
                    auroc,fpr=tmp_metric[0],tmp_metric[-1]
                    # print_str += '\n' +  f"{name2+'_f+o:10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"


                ood_metrics[name1] = np.array(ood_metrics[name1])
                mean_metric = ood_metrics[name1].mean(axis=0)
                auroc,fpr=mean_metric[0],mean_metric[-1]
                print_str += '\n' +  f"{name1:10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                print_str += '\n' + "--"*20
                
                ood_metrics[name1+'_r'] = np.array(ood_metrics[name1+'_r'])
                mean_metric = ood_metrics[name1+'_r'].mean(axis=0)
                auroc,fpr=mean_metric[0],mean_metric[-1]
                print_str += '\n' +  f"{name1+'-idr':10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                print_str += '\n' + "--"*20

                ood_metrics[name1+'_f'] = np.array(ood_metrics[name1+'_f'])
                mean_metric = ood_metrics[name1+'_f'].mean(axis=0)
                auroc,fpr=mean_metric[0],mean_metric[-1]
                print_str += '\n' +  f"{name1+'-idf':10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                print_str += '\n' + "--"*20

                ood_metrics[name1+'_f+o'] = np.array(ood_metrics[name1+'_f+o'])
                mean_metric = ood_metrics[name1+'_f+o'].mean(axis=0)
                auroc,fpr=mean_metric[0],mean_metric[-1]
                print_str += '\n' +  f"{name1+'+idf-idr':10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                print_str += '\n' + "--"*20

                # idr-idf
                tmp_score11 = tmp_score1[id_right_mask]
                tmp_label11 = tmp_label1[id_right_mask]
                tmp_score12 = tmp_score1[~id_right_mask]
                tmp_label12 = 0*tmp_label1[~id_right_mask]-1
                tmp_score = np.concatenate([tmp_score11,tmp_score12])
                tmp_label = np.concatenate([tmp_label11,tmp_label12])
                tmp_metric = auc_and_fpr_recall(tmp_score,tmp_label)
                ood_metrics['idr_idf'].append(tmp_metric)
                auroc,fpr=tmp_metric[0],tmp_metric[-1]
                print_str += '\n' +  f"{'idr_idf':10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                print_str += '\n' + "--"*20

                # np.set_printoptions(precision=4)
    print_str += '\n' + "#"+"=="*20
    print_str += '\n' + "#"+"==" + "fsood end"  if fsood else  '\n' + "#"+"==" + "general ood end"
    print_str += '\n' + "#"+"=="*20
    if IFSHOW:
        print(print_str)
    return print_str#feat_dict


def process_ood_metrics_openood_str2(feat_dict,fsood=False,IFSHOW=True):
    CSID_NAME_LIST = ['imagenet_c','imagenet_r','imagenet_v2']
    # OOD_FAR_LIST = ['inaturalist','openimage_o','textures']
    # OOD_NEAR_LIST = ['ninco','ssb_hard']
    # feat dict should contain the score
    print_str = ''
    # print_str += '\n' + "#"+"=="*20
    # print_str += '\n' + "#"+"==" + "fsood begin"  if fsood else  '\n' + "#"+"==" + "general ood begin"
    # print_str += '\n' + "#"+"=="*20
    # print_str += '%10s\t%5s\t%5s\t%5s' %('','auroc','fpr95','ID Acc')
    ret_dict = {}
    for tmp_score_name in list(feat_dict['id-test'].keys()):
        if not ('score' in tmp_score_name):
            continue 
        else:
            tmp_score_dict = {}
            # # construct printing str
            # print_str += '\n' + "=="*20
            # print_str += '\n' +  '%10s\t%5s\t%5s\t%5s' %(tmp_score_name,'auroc','fpr95','ID Acc')
            # print_str += '\n' + "--"*20

            # compute the acc
            tkey='id-test'
            tmp_label = feat_dict[tkey]['label_list']
            tmp_pred = feat_dict[tkey]['predcls']
            if fsood:
                csid_label = []
                csid_pred = []
                # combine the csid to id
                for csid_name in CSID_NAME_LIST:
                    csid_key = f"csid-{csid_name}"
                    csid_label.append(feat_dict[csid_key]['label_list'])
                    csid_pred.append(feat_dict[csid_key]['predcls'])
                csid_label = np.concatenate(csid_label,axis=0)
                csid_pred = np.concatenate(csid_pred,axis=0)
                tmp_label = np.concatenate((tmp_label,csid_label),axis=0)
                tmp_pred = np.concatenate((tmp_pred,csid_pred),axis=0)
    

            test_acc = acc(tmp_pred, tmp_label)
            tmp_score_dict['id_acc'] = test_acc
            # get the id right idx
            id_right_mask = (tmp_pred == tmp_label)
            # print(f'ID-{setkey}:{test_acc:.4f}')

            ood_metrics = {
                'far': [],
                'near': [],
                'far_r': [],
                'near_r': [],
                'far_f': [],
                'near_f': [],                
                'far_f+o': [],
                'near_f+o': [], 
                'idr_idf':[],
            }

            # setkey='test'
            set1 = feat_dict[tkey]
            tmp_score1 = set1[tmp_score_name]
            tmp_label1 = set1['label_list']
            if fsood:
                csid_label = []
                csid_score = []
                # combine the csid to id
                for csid_name in CSID_NAME_LIST:
                    csid_key = f"csid-{csid_name}"
                    csid_label.append(feat_dict[csid_key]['label_list'])
                    csid_score.append(feat_dict[csid_key][tmp_score_name])
                csid_label = np.concatenate(csid_label,axis=0)
                csid_score = np.concatenate(csid_score,axis=0)
                tmp_label1 = np.concatenate((tmp_label1,csid_label),axis=0)
                tmp_score1 = np.concatenate((tmp_score1,csid_score),axis=0)


            for name1 in ['far','near']:
                for tmp_key in list(feat_dict.keys()):
                    if not ('-'+name1+'-' in tmp_key):
                        continue
                    set2_key = tmp_key
                    set2 = feat_dict[set2_key]
                    # --this is the id_all and ood
                    tmp_score2 = set2[tmp_score_name]
                    tmp_label2 = -1 * np.ones_like(set2['label_list'])
                    tmp_score = np.concatenate([tmp_score1,tmp_score2])
                    tmp_label = np.concatenate([tmp_label1,tmp_label2])
                    tmp_metric = auc_and_fpr_recall(tmp_score,tmp_label)
                    #print(name1,name2,np.array(tmp_metric))
                    ood_metrics[name1].append(tmp_metric)
                    auroc,fpr=tmp_metric[0],tmp_metric[-1]
                    # print_str += '\n' +  f"{name2:10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                    # # -- this is the id-false and ood
                    # tmp_score11 = tmp_score1[id_right_mask]
                    # tmp_label11 = tmp_label1[id_right_mask]
                    # tmp_score = np.concatenate([tmp_score11,tmp_score2])
                    # tmp_label = np.concatenate([tmp_label11,tmp_label2])
                    # tmp_metric = auc_and_fpr_recall(tmp_score,tmp_label)                    
                    # ood_metrics[name1+'_r'].append(tmp_metric)
                    # auroc,fpr=tmp_metric[0],tmp_metric[-1]
                    # # print_str += '\n' +  f"{name2+'_r':10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"

                    # tmp_score11 = tmp_score1[~id_right_mask]
                    # tmp_label11 = tmp_label1[~id_right_mask]
                    # tmp_score = np.concatenate([tmp_score11,tmp_score2])
                    # tmp_label = np.concatenate([tmp_label11,tmp_label2])
                    # tmp_metric = auc_and_fpr_recall(tmp_score,tmp_label)
                    # ood_metrics[name1+'_f'].append(tmp_metric)
                    # auroc,fpr=tmp_metric[0],tmp_metric[-1]
                    # # print_str += '\n' +  f"{name2+'_f:10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"

                    # tmp_score11 = tmp_score1[id_right_mask]
                    # tmp_label11 = tmp_label1[id_right_mask]
                    # tmp_score12 = tmp_score1[~id_right_mask]
                    # tmp_label12 = 0*tmp_label1[~id_right_mask]-1
                    # tmp_score = np.concatenate([tmp_score11,tmp_score12,tmp_score2])
                    # tmp_label = np.concatenate([tmp_label11,tmp_label12,tmp_label2])
                    # tmp_metric = auc_and_fpr_recall(tmp_score,tmp_label)
                    # ood_metrics[name1+'_f+o'].append(tmp_metric)
                    # auroc,fpr=tmp_metric[0],tmp_metric[-1]
                    # # print_str += '\n' +  f"{name2+'_f+o:10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"


                ood_metrics[name1] = np.array(ood_metrics[name1])
                mean_metric = ood_metrics[name1].mean(axis=0)
                auroc,fpr=mean_metric[0],mean_metric[-1]
                # print_str += '\n' +  f"{name1:10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                # print_str += '\n' + "--"*20
                tmp_score_dict[name1+'_auroc'] = auroc
                tmp_score_dict[name1+'_fpr'] = fpr

                
                # ood_metrics[name1+'_r'] = np.array(ood_metrics[name1+'_r'])
                # mean_metric = ood_metrics[name1+'_r'].mean(axis=0)
                # auroc,fpr=mean_metric[0],mean_metric[-1]
                # print_str += '\n' +  f"{name1+'-idr':10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                # print_str += '\n' + "--"*20

                # ood_metrics[name1+'_f'] = np.array(ood_metrics[name1+'_f'])
                # mean_metric = ood_metrics[name1+'_f'].mean(axis=0)
                # auroc,fpr=mean_metric[0],mean_metric[-1]
                # print_str += '\n' +  f"{name1+'-idf':10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                # print_str += '\n' + "--"*20

                # ood_metrics[name1+'_f+o'] = np.array(ood_metrics[name1+'_f+o'])
                # mean_metric = ood_metrics[name1+'_f+o'].mean(axis=0)
                # auroc,fpr=mean_metric[0],mean_metric[-1]
                # print_str += '\n' +  f"{name1+'+idf-idr':10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                # print_str += '\n' + "--"*20

                # # idr-idf
                # tmp_score11 = tmp_score1[id_right_mask]
                # tmp_label11 = tmp_label1[id_right_mask]
                # tmp_score12 = tmp_score1[~id_right_mask]
                # tmp_label12 = 0*tmp_label1[~id_right_mask]-1
                # tmp_score = np.concatenate([tmp_score11,tmp_score12])
                # tmp_label = np.concatenate([tmp_label11,tmp_label12])
                # tmp_metric = auc_and_fpr_recall(tmp_score,tmp_label)
                # ood_metrics['idr_idf'].append(tmp_metric)
                # auroc,fpr=tmp_metric[0],tmp_metric[-1]
                # print_str += '\n' +  f"{'idr_idf':10s}\t{100*auroc:5.2f}\t{100*fpr:5.2f}\t{100*test_acc:5.2f}"
                # print_str += '\n' + "--"*20

                # np.set_printoptions(precision=4)

            # add things to the print_str
            # print_str += '\n' + "=="*20
            print_str +=   '%-10s\t%5s\t%5s\t%5s\t  %5s\t%5s\t%5s' %(tmp_score_name,'N-AUC','F-AUC','ID Acc','N-FPR','F-FPR','ID Acc') +'\n'
            print_str +=   f"{tmp_score_name:10s}\t{100*tmp_score_dict['near_auroc']:5.2f}\t{100*tmp_score_dict['far_auroc']:5.2f}\t{100*tmp_score_dict['id_acc']:5.2f}\t  {100*tmp_score_dict['near_fpr']:5.2f}\t{100*tmp_score_dict['far_fpr']:5.2f}\t{100*tmp_score_dict['id_acc']:5.2f}" +'\n'
            # print_str += '\n' + "=="*20
            ret_dict[tmp_score_name] = [100*tmp_score_dict['near_auroc'],100*tmp_score_dict['far_auroc'],100*tmp_score_dict['id_acc'],100*tmp_score_dict['near_fpr'],100*tmp_score_dict['far_fpr'],100*tmp_score_dict['id_acc']]
            
        
    # print_str += '\n' + "#"+"=="*20
    # print_str += '\n' + "#"+"==" + "fsood end"  if fsood else  '\n' + "#"+"==" + "general ood end"
    # print_str += '\n' + "#"+"=="*20
    if IFSHOW:
        print(print_str)
    return print_str,ret_dict#feat_dict

# ==============================================
# process the ood_score functions
# ==============================================
# compute the ood_score
def process_ood_scores(feat_dict,score_name,score_func,param_dict={},if_gpu=True,MAX_SIZE=1000,skip_train=False):
    # temperature = param_dict['temperature']
    # compute the multiple ood_scores
    for tkey in list(feat_dict.keys()):
        if ('id' in tkey) and ('train' in tkey) and skip_train:
            continue
        tmp_feat_dict = feat_dict[tkey]
        # split the feat_dict
        tmp_feat_dict_list = split_dict(tmp_feat_dict,MAXL=MAX_SIZE)
        # for splited_feat_dict in tmp_feat_dict_list:
        for splited_feat_dict in tqdm(tmp_feat_dict_list):
            tmp_eboscore = score_func(splited_feat_dict,param_dict=param_dict,if_gpu=if_gpu)
            if isinstance(tmp_eboscore,dict):
                for skey in list(tmp_eboscore.keys()):
                    splited_feat_dict[score_name+'-'+skey] = tmp_eboscore[skey].detach().numpy()
            else:
                splited_feat_dict[score_name] = tmp_eboscore.detach().numpy()
        feat_dict[tkey] = concat_dict(tmp_feat_dict_list)

    return feat_dict


# define a general post processor class
class base_ood_eval():
    def __init__(self,param_dict,search_dict={}):
        # pass # here will set some parameters
        self.param_dict = copy.deepcopy(param_dict) 
        self.search_dict = copy.deepcopy(search_dict) # each item is a list of param values
        # self.feat_dict = feat_dict
        self.hyperparam_search_done = False

    def get_param(self):
        return self.param_dict
    
    def set_param_dict(self,param_dict):
        if not (param_dict is None):
            self.param_dict.update(param_dict)

    def set_param_list(self,key_list,value_list):
        for idx,name in enumerate(key_list):
            self.param_dict[name] = value_list[idx]

    def hyperparam_search(self,feat_dict):
        self.feat_dict = feat_dict
        print('Starting automatic parameter search...')
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0

        for name in list(self.search_dict.keys()):
            hyperparam_names.append(name)
            count += 1

        for name in hyperparam_names:
            hyperparam_list.append(self.search_dict[name])
        # get the combination of all the parameters
        hyperparam_combination = self.recursive_generator(
            hyperparam_list, count)

        # search the best param use the validation set
        final_index = None
        for i, hyperparam in enumerate(hyperparam_combination):
            self.set_param_list(hyperparam_names,hyperparam)

            feat_dict = self.feat_dict['id-val']
            id_score = self.postprocess(feat_dict,param_dict=self.param_dict,if_gpu=True)
            feat_dict = self.feat_dict['ood-val']
            ood_score = self.postprocess(feat_dict,param_dict=self.param_dict,if_gpu=True)
            if isinstance(id_score,torch.Tensor):
                id_score = id_score.cpu().numpy()
            if isinstance(ood_score,torch.Tensor):
                ood_score = ood_score.cpu().numpy()
            
            id_gt = np.ones(id_score.shape,dtype=np.int32)
            ood_gt = -1*np.ones(ood_score.shape,dtype=np.int32)

            score = np.concatenate([id_score, ood_score])
            label = np.concatenate([id_gt, ood_gt])

            ood_metrics = auc_and_fpr_recall(score, label, tpr_th=0.95)
            auroc = ood_metrics[1]

            print('Hyperparam: {}, auroc: {}'.format(hyperparam, auroc))
            if auroc > max_auroc:
                final_index = i
                max_auroc = auroc

        self.set_param_list(hyperparam_names,hyperparam_combination[final_index])

        print('Final hyperparam: {}'.format(
            self.get_param()))
        self.hyperparam_search_done = True

    def recursive_generator(self, list, n):
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results

    def postprocess(self,feat_dict,param_dict={},if_gpu=True):
        pass

class FeatNormScore(base_ood_eval):
    def __init__(self,net=None,param_dict={'feat_th':1e8},search_dict={}):
        super().__init__(param_dict,search_dict)
        fc = get_fc(net)
        self.fc = fc#.clone()

    def postprocess(self,feat_dict,param_dict={'feat_th':1e8},if_gpu=True):
        feat = feat_dict['feat_list']
        feat_th = param_dict['feat_th']
        if isinstance(feat,np.ndarray):
            feat = torch.from_numpy(feat)
        if if_gpu:
            feat = feat.cuda()
            ood_score = torch.norm(feat.clamp(max=feat_th),p=2,dim=-1)
            feat = feat.cpu()
            ood_score = ood_score.cpu().detach()
        else:
            ood_score = torch.norm(feat.clamp(max=feat_th),p=2,dim=-1)
            ood_score = ood_score.cpu().detach()
        return ood_score


# this is "Exploiting Discrepancy in Feature Statistic"
class FSEnergyScore(base_ood_eval):
    def __init__(self,net=None,param_dict={'clip_th':1},search_dict={}):
        super().__init__(param_dict,search_dict)
        fc = get_fc(net)
        self.fc = fc#.clone()

    def postprocess(self,feat_dict,param_dict={'clip_th':1},if_gpu=True):
        clip_th = param_dict['clip_th']
        feat = feat_dict['feat_list']
        if isinstance(feat,np.ndarray):
            feat = torch.from_numpy(feat)
        if if_gpu:
            feat = feat.cuda()
            self.fc = self.fc.cuda()
            logits = self.fc(feat.clamp(max=clip_th))
            ood_score = (1+feat.mean(dim=-1))*torch.logsumexp(logits, dim=1)
            feat = feat.cpu()
            self.fc = self.fc.to('cpu')
            ood_score = ood_score.cpu().detach()
        else:
            logits = self.fc(feat.clamp(max=clip_th))
            ood_score = (1+feat.mean(dim=-1))*torch.logsumexp(logits, dim=1)
            ood_score = ood_score.cpu().detach()
        return ood_score

class EnergyScore(base_ood_eval):
    def __init__(self,param_dict={'temperature':1},search_dict={}):
        super().__init__(param_dict,search_dict)

    def postprocess(self,feat_dict,param_dict={'temperature':1},if_gpu=True):
        temperature = param_dict['temperature']
        logits = feat_dict['logit_list']
        if isinstance(logits,np.ndarray):
            logits = torch.from_numpy(logits)
        if if_gpu:
            ood_score = temperature*torch.logsumexp(logits.cuda()/temperature,dim=1)
            ood_score = ood_score.cpu()
        else:
            ood_score = temperature*torch.logsumexp(logits/temperature,dim=1)
        return ood_score


class MaxLogitScore(base_ood_eval):
    def __init__(self,param_dict={},search_dict={}):
        super().__init__(param_dict,search_dict)

    def postprocess(self,feat_dict,param_dict={},if_gpu=True):
        logits = feat_dict['logit_list']
        if isinstance(logits,np.ndarray):
            logits = torch.from_numpy(logits)
        if if_gpu:
            ood_score = logits.cuda().max(dim=1)[0].cpu()
        else:
            ood_score = logits.max(dim=1)[0] # get the maximum logits
        return ood_score


class MaxSoftmaxScore(base_ood_eval):
    def __init__(self,param_dict={'temperature':1},search_dict={}):
        super().__init__(param_dict,search_dict)    

    def postprocess(self,feat_dict,param_dict={'temperature':1},if_gpu=True):
        logits = feat_dict['logit_list']
        temperature = param_dict['temperature']
        if isinstance(logits,np.ndarray):
            logits = torch.from_numpy(logits)
        if if_gpu:
            logits_softmax = torch.softmax(logits.cuda()/temperature,dim=1)
            ood_score = logits_softmax.max(dim=1)[0].cpu() # get the maximum logits
        else:
            logits_softmax = torch.softmax(logits/temperature,dim=1)
            ood_score = logits_softmax.max(dim=1)[0] # get the maximum logits
        return ood_score

# save the minimal percentage, fill the same value
def ash_b(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x

# save the minimal percentage, fill the original value
def ash_p(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100

    b, c, h, w = x.shape

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    return x

# save the minimal percentage, fill the original value and scale
def ash_s(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x


def ash_rand(x, percentile=65, r1=0, r2=10):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    v = v.uniform_(r1, r2)
    t.zero_().scatter_(dim=1, index=i, src=v)
    return x

# only do scale
def ash_scale(x, percentile=65):
    input = x.clone()
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    
    return input * torch.exp(scale[:, None, None, None])


class ASHScore(base_ood_eval):
    def __init__(self,net,param_dict={'percentile':90,'mod':'ash_b'},search_dict={}):
        super().__init__(param_dict,search_dict)    
        # get the fc layer of the net
        fc = get_fc(net)
        self.fc = fc#.clone()
        self.mod = param_dict['mod']
        if self.mod == 'ash_b':
            self.func = ash_b
        elif self.mod == 'ash_p':
            self.func = ash_p 
        elif self.mod == 'ash_s':
            self.func = ash_s 
        elif self.mod == 'ash_rand':
            self.func = ash_rand
        elif self.mod == 'ash_scale':
            self.func = ash_scale             

    def postprocess(self,feat_dict,param_dict={'percentile':90},if_gpu=True):
        feat = feat_dict['feat_list']
        if isinstance(feat,np.ndarray):
            feat = torch.from_numpy(feat)
        if if_gpu:
            feat = feat.cuda()
            self.fc = self.fc.cuda()
            # change the size of the feature
            feat_size = feat.size()
            feat_size_len = len(feat_size)
            if feat_size_len==2:
                feat = feat[:,:,None,None]
            feat = self.func(feat,percentile=param_dict['percentile'])
            # reshape the size of feature
            feat = feat.reshape(feat_size)
            
            logits = self.fc(feat)
            ood_score = torch.logsumexp(logits, dim=1)
            feat = feat.cpu()
            self.fc = self.fc.to('cpu')
            ood_score = ood_score.cpu().detach()
        else:
            feat = self.func(feat,percentile=param_dict['percentile'])
            logits = self.fc(feat)
            ood_score = torch.logsumexp(logits, dim=1)
            ood_score = ood_score.detach()
        return ood_score


# compute the temperature scaling
class TemperatureScaling(base_ood_eval):
    """A decorator which wraps a model with temperature scaling, internalize
    'temperature' parameter as part of a net model."""
    def __init__(self,param_dict={},search_dict={}):
        super().__init__(param_dict={},search_dict={})
        self.temperature = nn.Parameter(torch.ones(1, device='cuda') *
                                        1.5)  # initialize T as 1.5
        self.setup_flag = False # flag for parameters tuned

    def setup(self, val_dict):
        nll_criterion = nn.CrossEntropyLoss().cuda()

        with torch.no_grad(
        ):  # fix other params of the net, only learn temperature
            logits = torch.from_numpy(val_dict['logit_list']).float().cuda()
            labels = torch.from_numpy(val_dict['label_list']).cuda()
            # calculate NLL before temperature scaling
            before_temperature_nll = nll_criterion(logits, labels)

        print('Before temperature - NLL: %.3f' % (before_temperature_nll))

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        # make sure only temperature parameter will be learned,
        # fix other parameters of the network
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self._temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # print learned parameter temperature,
        # calculate NLL after temperature scaling
        after_temperature_nll = nll_criterion(
            self._temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f' % (after_temperature_nll))
        self.setup_flag = True


    def _temperature_scale(self, logits):
        return logits / self.temperature

    @torch.no_grad()
    def postprocess(self, feat_dict,param_dict={},if_gpu=True):
        logits = feat_dict['logit_list']
        if isinstance(logits,np.ndarray):
            logits = torch.from_numpy(logits)
        if if_gpu:
            logits_ts = logits.cuda() / self.temperature.cuda()
            score = torch.softmax(logits_ts, dim=1)
            conf, _ = torch.max(score, dim=1)
            conf = conf.cpu()
        else:
            logits_ts = logits.cpu() / self.temperature.cpu()
            score = torch.softmax(logits_ts, dim=1)
            conf, _ = torch.max(score, dim=1)
        return  conf

class LeverageImportNeuron(base_ood_eval):
    def __init__(self,net,cmat_path,param_dict={'clip_th':0.8,'pa':10,'pw':10},search_dict={}):
        super().__init__(param_dict,search_dict)
        # get the fc layer of the net
        fc = get_fc(net)
        self.weight = fc.weight.cpu().clone()
        self.bias = fc.bias.cpu().clone()
        # load the contribution matrix 
        cmat = np.load(cmat_path)
        cmat = cmat.T # change to cnum*feat_dim
        self.cmat = cmat
        # initilize the weight and feature mask
        self.masked_w = None
        self.mask_f = None
        self.in_dim = self.weight.size()[1]
        self.out_dim = self.weight.size()[0]
        self.pa = param_dict['pa']
        self.pw = param_dict['pw']

    # this no need run every time
    def calculate_shap_value(self,param_dict={'clip_th':0.8,'pa':10,'pw':10}):
        self.mask_f = torch.zeros(self.out_dim,self.in_dim)
        self.masked_w = torch.zeros((self.out_dim,self.out_dim,self.in_dim))
        pa = param_dict['pa']
        pw = param_dict['pw']

        for class_num in range(self.out_dim):
            self.matrix = abs(self.cmat[class_num,:].reshape(1,-1)) * self.weight.data.cpu().numpy()
            self.thresh = np.percentile(self.matrix, pw)
            mask_w = torch.Tensor((self.matrix > self.thresh))
            self.masked_w[class_num,:,:] = (self.weight.squeeze().cpu() * mask_w).cuda()
            self.class_thresh = np.percentile(self.cmat[class_num,:], pa)
            self.mask_f[class_num,:] = torch.Tensor((self.cmat[class_num,:] > self.class_thresh))
        # self.clip_th = param_dict['clip_th']
        self.pa = param_dict['pa']
        self.pw = param_dict['pw']


    def postprocess(self, feat_dict,param_dict={'clip_th':0.8,'pa':10,'pw':10},if_gpu=True):
        with torch.no_grad():
            features = feat_dict['feat_list']
            if isinstance(features,np.ndarray):
                features = torch.from_numpy(features) 

            if if_gpu:
                features = features.cuda()
                self.weight = self.weight.cuda()
                if self.bias is not None:
                    self.bias = self.bias.cuda()
            # do feature clip
            clip_th = param_dict['clip_th']
            features = features.clip(max=clip_th)       
            # compute the feature mask and updated weight

            
            if (self.pa != param_dict['pa']) or (self.pw != param_dict['pw']):
                self.calculate_shap_value(param_dict)
            elif self.masked_w is None:
                self.calculate_shap_value(param_dict)
            # do the forward to get the prediction
            pred = (features[:, None, :] * self.weight.data[None,:,:]).sum(2)
            # pred = pred.cpu()
            if self.bias is not None:
                pred = pred + self.bias[None,:]
            else:
                pred = pred
            pred = torch.nn.functional.softmax(pred, dim=1)   
            pred = np.argmax(pred.cpu().detach().numpy(), axis=1)     
            
            

            if if_gpu:

                # here is too slow, will make an efficient version
                features = features.cuda()
                # do activation clipping
                counter_cp = 0
                cp = torch.zeros((features.size()[0], self.in_dim)).cuda()
                out = torch.zeros((len(pred),self.out_dim)).cuda()
                # can do togther for all_idx
                unique_pred = np.unique(pred)
                for pred_v in unique_pred:
                    pidx = (pred==pred_v)
                    cp[pidx,:] = features[pidx,:].cuda()   * self.mask_f[pred_v,:].cuda()
                    out[pidx,:] = (cp[pidx,:][:,None,:].cuda()  * self.masked_w[pred_v,:,:][None,:,:].cuda()  ).sum(2)
                # for idx in pred:
                #     cp[counter_cp,:] = features[counter_cp,:].cuda()   * self.mask_f[idx,:].cuda()  
                #     out[counter_cp,:] = (cp[counter_cp,:][None,None,:].cuda()  * self.masked_w[idx,:,:][None,:,:].cuda()  ).sum(2)
                #     counter_cp = counter_cp + 1


                # # here is too slow, will make an efficient version
                # features = features.cuda()
                # # do activation clipping
                # counter_cp = 0
                # cp = torch.zeros((features.size()[0], self.in_dim)).cuda()
                # for idx in pred:
                #     cp[counter_cp,:] = features[counter_cp,:].cuda()   * self.mask_f[idx,:].cuda()  
                #     counter_cp = counter_cp + 1

                # out = torch.zeros((len(pred),self.out_dim)).cuda()
                # counter_dice = 0
                # for idx in pred:
                #     out[counter_dice,:] = (cp[counter_dice,:][None,None,:].cuda()  * self.masked_w[idx,:,:][None,:,:].cuda()  ).sum(2)
                #     counter_dice = counter_dice + 1
                
                
                if self.bias is not None:
                    out = out + self.bias[None,:]
                else:
                    out = out
                out =  torch.logsumexp(out, dim=1)
                out = out.cpu()
                features = features.cpu()
                self.weight = self.weight.cpu()
                if self.bias is not None:
                    self.bias = self.bias.cpu()

            else:
                # do activation clipping
                counter_cp = 0
                cp = torch.zeros((features.size()[0], self.in_dim))
                for idx in pred:
                    cp[counter_cp,:] = features[counter_cp,:] * self.mask_f[idx,:]  
                    counter_cp = counter_cp + 1

                out = torch.zeros((len(pred),self.out_dim))
                counter_dice = 0
                for idx in pred:
                    out[counter_dice,:] = (cp[counter_dice,:][None,None,:] * self.masked_w[idx,:,:][None,:,:]).sum(2)
                    counter_dice = counter_dice + 1
                
                if self.bias is not None:
                    out = out + self.bias[None,:]
                else:
                    out = out                    
                    out =  torch.logsumexp(out, dim=1)#.detach().cpu().numpy()
        return out



# below is the kn postporcessor
normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
class KNNPostprocessor(base_ood_eval):
    def __init__(self, param_dict={'K':50,'alpha':0.1},search_dict={}):
        super().__init__(param_dict,search_dict) 
        self.setup_flag = False
        # self.args = self.config.postprocessor.postprocessor_args
        # self.activation_log = None
        # self.args_dict = self.config.postprocessor.postprocessor_sweep
        

    def setup(self, train_features):
        if not self.setup_flag:
            # added sample some sample
            alpha = self.param_dict['alpha']
            flen = train_features.shape[0]
            slen = int(alpha*flen)
            if alpha<1:
                np.random.seed(95)
                new_index = np.random.permutation(flen)
                new_index = new_index[:slen]
                train_features = train_features[new_index,:]           
            # faiss only accept float32
            self.activation_log = normalizer(train_features).astype('float32') 
            self.index = faiss.IndexFlatL2(train_features.shape[1])
            self.index.add(self.activation_log)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self,feat_dict,param_dict={'K':50},if_gpu=True):
        feature = feat_dict['feat_list']
        feature_normed = normalizer(feature).astype('float32')
        D, _ = self.index.search(
            feature_normed,
            param_dict['K'],
        )
        # print(D) # D is from small to high
        kth_dist = -D[:, -1] #this is only the distance from 1 element, the K-th elements
        # kth_dist = -np.array(D.mean(axis=1))
        return torch.from_numpy(kth_dist)

    def set_hyperparam(self, param_dict={'K':50}):
        self.param_dict['K'] = param_dict['K']

    def get_hyperparam(self):
        return self.param_dict['K']

# below is the kn postporcessor
normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
class KNNMeanPostprocessor(base_ood_eval):
    def __init__(self, param_dict={'K':50,'alpha':0.1},search_dict={}):
        super().__init__(param_dict,search_dict) 
        self.setup_flag = False
        # self.args = self.config.postprocessor.postprocessor_args
        # self.activation_log = None
        # self.args_dict = self.config.postprocessor.postprocessor_sweep
        

    def setup(self, train_features):
        if not self.setup_flag:
            # added sample some sample
            alpha = self.param_dict['alpha']
            flen = train_features.shape[0]
            slen = int(alpha*flen)
            if alpha<1:
                np.random.seed(95)
                new_index = np.random.permutation(flen)
                new_index = new_index[:slen]
                train_features = train_features[new_index,:]           
            # faiss only accept float32
            self.activation_log = normalizer(train_features).astype('float32') 
            self.index = faiss.IndexFlatL2(train_features.shape[1])
            self.index.add(self.activation_log)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self,feat_dict,param_dict={'K':50},if_gpu=True):
        feature = feat_dict['feat_list']
        feature_normed = normalizer(feature).astype('float32')
        D, _ = self.index.search(
            feature_normed,
            param_dict['K'],
        )
        # print(D) # D is from small to high
        # kth_dist = -D[:, -1] #this is only the distance from 1 element, the K-th elements
        kth_dist = -np.array(D.mean(axis=-1))
        return torch.from_numpy(kth_dist)

    def set_hyperparam(self, param_dict={'K':50}):
        self.param_dict['K'] = param_dict['K']

    def get_hyperparam(self):
        return self.param_dict['K']

# compute the NNguide score
def knn_score(bankfeas, queryfeas, k=100, min=False):

    # bankfeas = deepcopy(np.array(bankfeas))
    # queryfeas = deepcopy(np.array(queryfeas))

    index = faiss.IndexFlatIP(bankfeas.shape[-1])
    index.add(bankfeas)
    D, _ = index.search(queryfeas, k)
    if min:
        scores = np.array(D.min(axis=1))
    else:
        scores = np.array(D.mean(axis=1))
    return scores # this is different from KNN score, which use k-th result


class NNGuidePostprocessor(base_ood_eval):
    def __init__(self, param_dict={'K':100,'alpha':0.01},search_dict={}):
        super().__init__(param_dict,search_dict) 
        self.setup_flag = False

    def setup(self, bank_feas, bank_logits):
        alpha = self.param_dict['alpha']
        if not self.setup_flag:
            flen = bank_feas.shape[0]
            slen = int(alpha*flen)
            np.random.seed(95)
            new_index = np.random.permutation(flen)
            bank_feas = bank_feas[new_index,:]
            bank_logits = bank_logits[new_index,:]
            # bank_feas = np.concatenate(bank_feas, axis=0)
            bank_confs = logsumexp(bank_logits[:slen,:],axis=-1)
            # print(bank_feas.shape)
            # print(bank_confs.shape)
            self.bank_guide = bank_feas[:slen,:] * bank_confs[:, None]
            self.bank_guide = self.bank_guide.astype('float32')
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, feat_dict,param_dict={'K':100},if_gpu=True ):
        K = param_dict['K']
        feature = feat_dict['feat_list']
        feas_norm = normalizer(feature)
        feas_norm = feas_norm.astype('float32')
        conf = knn_score(self.bank_guide, feas_norm, k=K)
        logit = feat_dict['logit_list']
        energy = logsumexp(logit, axis=-1)
        score = conf * energy
        

        return torch.from_numpy(score) # here score is numpy

    def set_hyperparam(self, param_dict={'K':100,'alpha':0.01}):
        self.param_dict['K'] = param_dict['K']
        self.param_dict['alpha'] = param_dict['alpha']

    def get_hyperparam(self):
        K = self.param_dict['K']
        alpha = self.param_dict['alpha']
        return K #[K,alpha]

class MDSPostprocessor(base_ood_eval):
    def __init__(self, CNUM=100,param_dict={},search_dict={}):
        super().__init__(param_dict,search_dict) 
        self.CNUM = CNUM
        self.setup_flag = False

    def setup(self, features, labels):
        if not self.setup_flag:
            self.class_mean = []
            centered_data = []
            labels = np.squeeze(labels)
            for c in range(self.CNUM):
                class_samples = features[labels==c]#.data
                self.class_mean.append(class_samples.mean(axis=0, keepdims=True))
                # print(self.class_mean)
                centered_data.append(class_samples -
                                     self.class_mean[c].reshape(1, -1))                
            # self.class_mean = torch.stack(
            #     self.class_mean)  # shape [#classes, feature dim]
            self.class_mean = np.concatenate(
                self.class_mean,axis=0)  # shape [#classes, feature dim]
            centered_data = np.concatenate(centered_data,axis=0).astype(np.float32)

            group_lasso = sklearn.covariance.EmpiricalCovariance(
                assume_centered=False)
            # group_lasso.fit(
            #     torch.cat(centered_data).cpu().numpy().astype(np.float32))
            group_lasso.fit(
                centered_data)
            # inverse of covariance
            self.class_mean = torch.from_numpy(self.class_mean).float()
            self.precision = torch.from_numpy(group_lasso.precision_).float()
            self.setup_flag = True
        else:
            pass

    # @torch.no_grad()
    # def postprocess(self, feat_dict,param_dict={'K':100},if_gpu=True ):
    #     feature = feat_dict['feat_list']
    #     feature = torch.from_numpy(feature).float()
    #     class_scores = torch.zeros((feature.shape[0], self.CNUM))
    #     for c in range(self.CNUM):
    #         tensor = feature.cpu() - self.class_mean[c].view(1, -1)
    #         class_scores[:, c] = -torch.matmul(
    #             torch.matmul(tensor, self.precision), tensor.t()).diag()
    #         # class_scores[:, c] = (-torch.matmul(
    #         #     torch.matmul(tensor.cuda(), self.precision.cuda()), tensor.t().cuda()).diag()).cpu()

    #     score = torch.max(class_scores, dim=1)[0]
    #     return score # here score is numpy

    @torch.no_grad()
    def postprocess(self, feat_dict,param_dict={'K':100},if_gpu=True ):
        feature = feat_dict['feat_list']
        feature = torch.from_numpy(feature).float()
        class_scores = torch.zeros((feature.shape[0], self.CNUM))
        for c in range(self.CNUM):
            if not if_gpu:
                tensor = feature.cpu() - self.class_mean[c].view(1, -1)
                class_scores[:, c] = -torch.matmul(
                    torch.matmul(tensor, self.precision), tensor.t()).diag()
            else:
                tensor = feature.cuda() - self.class_mean[c].view(1, -1).cuda()
                class_scores[:, c] = (-torch.matmul(
                    torch.matmul(tensor.cuda(), self.precision.cuda()), tensor.cuda().t()).diag()).cpu()
            # class_scores[:, c] = (-torch.matmul(
            #     torch.matmul(tensor.cuda(), self.precision.cuda()), tensor.t().cuda()).diag()).cpu()

        score = torch.max(class_scores, dim=1)[0]
        return score # here score is numpy


class RMDSPostprocessor(base_ood_eval):
    def __init__(self, CNUM=100,param_dict={},search_dict={}):
        super().__init__(param_dict,search_dict) 
        self.CNUM = CNUM
        self.setup_flag = False

    def setup(self, features, labels):
        if not self.setup_flag:
            self.class_mean = []
            centered_data = []
            labels = np.squeeze(labels)
            for c in range(self.CNUM):
                class_samples = features[labels==c]#.data
                self.class_mean.append(class_samples.mean(axis=0, keepdims=True))
                centered_data.append(class_samples -
                                     self.class_mean[c].reshape(1, -1))                
            # self.class_mean = torch.stack(
            #     self.class_mean)  # shape [#classes, feature dim]
            self.class_mean = np.concatenate(
                self.class_mean,axis=0)  # shape [#classes, feature dim]
            centered_data = np.concatenate(centered_data,axis=0).astype(np.float32)

            group_lasso = sklearn.covariance.EmpiricalCovariance(
                assume_centered=False)
            # group_lasso.fit(
            #     torch.cat(centered_data).cpu().numpy().astype(np.float32))
            group_lasso.fit(
                centered_data)
            # inverse of covariance
            self.class_mean = torch.from_numpy(self.class_mean).float()
            self.precision = torch.from_numpy(group_lasso.precision_).float()
            self.setup_flag = True

            # self.whole_mean = features.mean(0)
            # centered_data = features - self.whole_mean.view(1, -1)
            self.whole_mean = features.mean(axis=0, keepdims=True)
            centered_data = features - self.whole_mean.reshape(1, -1)
            group_lasso = sklearn.covariance.EmpiricalCovariance(
                assume_centered=False)
            # group_lasso.fit(centered_data.cpu().numpy().astype(np.float32))
            group_lasso.fit(centered_data.astype(np.float32))
            self.whole_mean = torch.from_numpy(self.whole_mean).float()
            self.whole_precision = torch.from_numpy(
                group_lasso.precision_).float()

        else:
            pass

    # @torch.no_grad()
    # def postprocess(self, feat_dict,param_dict={'K':100},if_gpu=True ):
    #     feature = feat_dict['feat_list']
    #     feature = torch.from_numpy(feature).float()


    #     tensor1 = feature.cpu() - self.whole_mean.view(1, -1)
    #     background_scores = -torch.matmul(
    #         torch.matmul(tensor1, self.whole_precision), tensor1.t()).diag()
    #     # background_scores = (-torch.matmul(
    #     #     torch.matmul(tensor1.cuda(), self.whole_precision.cuda()), tensor1.t().cuda()).diag()).cpu()

    #     class_scores = torch.zeros((feature.shape[0], self.CNUM))
    #     for c in range(self.CNUM):
    #         tensor = feature.cpu() - self.class_mean[c].view(1, -1)
    #         class_scores[:, c] = -torch.matmul(
    #             torch.matmul(tensor, self.precision), tensor.t()).diag()
    #         # class_scores[:, c] = (-torch.matmul(
    #         #     torch.matmul(tensor.cuda(), self.precision.cuda()), tensor.t().cuda()).diag()).cpu()
    #         class_scores[:, c] = class_scores[:, c] - background_scores
        
    #     score = torch.max(class_scores, dim=1)[0]
    #     return score # here score is numpy

    @torch.no_grad()
    def postprocess(self, feat_dict,param_dict={'K':100},if_gpu=True ):
        feature = feat_dict['feat_list']
        feature = torch.from_numpy(feature).float()

        if if_gpu:
            tensor1 = feature.cuda() - self.whole_mean.view(1, -1).cuda()
            background_scores = -torch.matmul(
                torch.matmul(tensor1, self.whole_precision.cuda()), tensor1.t()).diag()
            del tensor1

            class_scores = torch.zeros((feature.shape[0], self.CNUM)).cuda()
            for c in range(self.CNUM):
                tensor = feature.cuda() - self.class_mean[c].view(1, -1).cuda()
                class_scores[:, c] = -torch.matmul(
                    torch.matmul(tensor, self.precision.cuda()), tensor.t()).diag()
                class_scores[:, c] = class_scores[:, c] - background_scores
            
            score = torch.max(class_scores, dim=1)[0]
            score = score.cpu()
            del class_scores
        else:
            tensor1 = feature.cpu() - self.whole_mean.view(1, -1)
            background_scores = -torch.matmul(
                torch.matmul(tensor1, self.whole_precision), tensor1.t()).diag()
            # background_scores = (-torch.matmul(
            #     torch.matmul(tensor1.cuda(), self.whole_precision.cuda()), tensor1.t().cuda()).diag()).cpu()

            class_scores = torch.zeros((feature.shape[0], self.CNUM))
            for c in range(self.CNUM):
                tensor = feature.cpu() - self.class_mean[c].view(1, -1)
                class_scores[:, c] = -torch.matmul(
                    torch.matmul(tensor, self.precision), tensor.t()).diag()
                # class_scores[:, c] = (-torch.matmul(
                #     torch.matmul(tensor.cuda(), self.precision.cuda()), tensor.t().cuda()).diag()).cpu()
                class_scores[:, c] = class_scores[:, c] - background_scores
            
            score = torch.max(class_scores, dim=1)[0]
        return score # here score is numpy


class GENPostprocessor(base_ood_eval):
    def __init__(self, param_dict={'gamma':0.1,'M':100},search_dict={}):
        super().__init__(param_dict,search_dict) 
        self.setup_flag = False


    @torch.no_grad()
    def postprocess(self, feat_dict,param_dict={'K':100},if_gpu=True ):
        if if_gpu:
            output = torch.from_numpy(feat_dict['logit_list']).cuda()
        else:
            output = torch.from_numpy(feat_dict['logit_list'])
        score = torch.softmax(output, dim=1)
        conf = self.generalized_entropy(score, self.param_dict['gamma'], self.param_dict['M'])
        if if_gpu:
            conf = conf.cpu()
        return conf        

        # return torch.from_numpy(score) # here score is numpy

    def set_hyperparam(self, param_dict={'gamma':0.1,'M':100}): # it seems not been used
        self.param_dict['gamma'] = param_dict['gamma']
        self.param_dict['M'] = param_dict['M']

    def get_hyperparam(self): # it seems not been used
        K = self.param_dict['gamma']
        alpha = self.param_dict['M']
        return [K,alpha]

    def generalized_entropy(self, softmax_id_val, gamma=0.1, M=100):
        probs = softmax_id_val
        probs_sorted = torch.sort(probs, dim=1)[0][:, -M:]
        scores = torch.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma),
                           dim=1)

        return -scores


class VIMPostprocessor(base_ood_eval):
    def __init__(self, param_dict={'dim':256},search_dict={}):
        super().__init__(param_dict,search_dict) 
        self.setup_flag = False

    def setup(self, net,features, logits):
        if not self.setup_flag:
            net.eval()
            # this step involves extract feature from the whole training set, which is very time consuming
            with torch.no_grad():
                self.w, self.b = net.get_fc()

            self.u = -np.matmul(pinv(self.w), self.b) # this is the bias
            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(features - self.u)
            eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            if ec.covariance_.shape[0]<self.param_dict['dim']:
                self.param_dict['dim'] = 256
            self.NS = np.ascontiguousarray(
                (eigen_vectors.T[np.argsort(eig_vals * -1)[self.param_dict['dim'] ] ]).T) # C*self.dim

            vlogit_id_train = norm(np.matmul(features - self.u,
                                             self.NS),
                                   axis=-1) # N*C * C*self.dim
            self.alpha = logits.max(
                axis=-1).mean() / vlogit_id_train.mean()
            print(f'{self.alpha=:.4f}')

            # change self.u self.NS self.alpha to tensor
            self.u = torch.from_numpy(self.u).float()
            self.NS = torch.from_numpy(self.NS).float()
            self.alpha = self.alpha.item()

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, feat_dict,param_dict={'K':100},if_gpu=True ):
        feature = feat_dict['feat_list']
        logit = feat_dict['logit_list']
        feature = torch.from_numpy(feature).float()
        logit = torch.from_numpy(logit).float()
        if not if_gpu:
            energy_ood = torch.logsumexp(logit, dim=-1) 
            vlogit_ood = torch.linalg.norm(np.matmul(feature - self.u, self.NS),
                            dim=-1) * self.alpha
            score_ood = -vlogit_ood + energy_ood
        else:
            energy_ood = torch.logsumexp(logit.cuda(), dim=-1) 
            vlogit_ood = torch.linalg.norm(torch.matmul(feature.cuda() - self.u.cuda(), self.NS.cuda()),
                            dim=-1) * self.alpha
            score_ood = -vlogit_ood + energy_ood
            score_ood = score_ood.cpu()
        return score_ood        

    def set_hyperparam(self, param_dict={'dim':256}):
        self.param_dict['dim'] = param_dict['dim']

    def get_hyperparam(self):
        K = self.param_dict['dim']
        return K #[K,alpha]


# class VIMPostprocessor(base_ood_eval):
#     def __init__(self, param_dict={'dim':256},search_dict={}):
#         super().__init__(param_dict,search_dict) 
#         self.setup_flag = False

#     def setup(self, net,features, logits):
#         if not self.setup_flag:
#             net.eval()
#             # this step involves extract feature from the whole training set, which is very time consuming
#             with torch.no_grad():
#                 self.w, self.b = net.get_fc()

#             self.u = -np.matmul(pinv(self.w), self.b) # this is the bias
#             ec = EmpiricalCovariance(assume_centered=True)
#             ec.fit(features - self.u)
#             eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
#             if ec.covariance_.shape[0]>1500:
#                 self.param_dict['dim']=1000
#             else:
#                 self.param_dict['dim']=256
#             self.eigen_vectors = eigen_vectors
#             self.eig_vals = eig_vals

#             self.setup_flag = True
#         else:
#             pass

#     @torch.no_grad()
#     def postprocess(self, feat_dict,param_dict={'K':100},if_gpu=True ):
#         self.NS = np.ascontiguousarray(
#             (self.eigen_vectors.T[np.argsort(self.eig_vals * -1)[self.param_dict['dim'] ] ]).T) # C*self.dim

#         vlogit_id_train = norm(np.matmul(features - self.u,
#                                             self.NS),
#                                 axis=-1) # N*C * C*self.dim
#         self.alpha = logits.max(
#             axis=-1).mean() / vlogit_id_train.mean()
#         print(f'{self.alpha=:.4f}')

#         # change self.u self.NS self.alpha to tensor
#         self.u = torch.from_numpy(self.u).float()
#         self.NS = torch.from_numpy(self.NS).float()
#         self.alpha = torch.from_numpy(self.alpha).float()        
        
        
        
#         feature = feat_dict['feat_list']
#         logit = feat_dict['logit_list']
#         feature = torch.from_numpy(feature).float()
#         logit = torch.from_numpy(logit).float()
#         if not if_gpu:
#             energy_ood = torch.logsumexp(logit, dim=-1) 
#             vlogit_ood = torch.linalg.norm(np.matmul(feature - self.u, self.NS),
#                             dim=-1) * self.alpha
#             score_ood = -vlogit_ood + energy_ood
#         else:
#             energy_ood = torch.logsumexp(logit.cuda(), dim=-1) 
#             vlogit_ood = torch.linalg.norm(np.matmul(feature.cuda() - self.u.cuda(), self.NS.cuda()),
#                             dim=-1) * self.alpha.cuda()
#             score_ood = -vlogit_ood + energy_ood
#             score_ood = score_ood.cpu()
#         return score_ood        

#     def set_hyperparam(self, param_dict={'dim':256}):
#         self.param_dict['dim'] = param_dict['dim']

#     def get_hyperparam(self):
#         K = self.param_dict['dim']
#         return K #[K,alpha]




if __name__ == '__main__':
    TS = TemperatureScaling()
    print(TS)

    # import time
    # start = time.time()
    # KNN = KNNPostprocessor(param_dict={'K':50},search_dict={'K':[50, 100, 200, 500, 1000]})
    # train_features = np.random.rand(1000*1000,50*10)
    # KNN.setup(train_features)
    # query_feature = np.random.rand(50000,50*10)
    # query_featdict ={'feat_list':query_feature}
    # KNNScore = KNN.postprocess(query_featdict,param_dict=KNN.param_dict)
    # end = time.time()
    # print(KNNScore)
    # print(end-start)

    import time
    start = time.time()
    train_features = np.random.rand(1000*1000,50*10)
    train_logits = np.random.rand(1000*1000,10)
    query_feature = np.random.rand(50000,50*10)
    query_logits = np.random.rand(50000,10)
    query_featdict ={'feat_list':query_feature,'logit_list':query_logits}
    NNGUIDE = NNGuidePostprocessor()
    NNGUIDE.setup(train_features, train_logits)
    NNGUIDEScore = NNGUIDE.postprocess(query_featdict,param_dict=NNGUIDE.param_dict)
    end = time.time()
    print(NNGUIDEScore)
    print(end-start)   