from torch.utils.data import Dataset
import torch
import numpy as np


class FeatDataset(Dataset):
    def __init__(self, feat, labels,label_tranmat=None):
        self.data = feat
        self.labels = labels
        self.len = feat.shape[0]
        assert self.len == len(labels)
        self.label_tranmat = label_tranmat
        if isinstance(self.label_tranmat,np.ndarray):
            self.label_tranmat = torch.from_numpy(self.label_tranmat).long()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        # need to change things to tensor
        if isinstance(data,np.ndarray):
            data = torch.from_numpy(data)
        if not isinstance(label,torch.Tensor):
            if isinstance(label,np.ndarray):
                label = torch.from_numpy(label)
            elif len(label.shape)==0:
                # label = label.item()
                label = torch.Tensor([label]).long()[:]
        # if len(label.size())==1:
        #     label = label[None,:]
        if isinstance(self.label_tranmat,torch.Tensor):
            label = self.label_tranmat[label,:]
            if len(label.size())>1:
                label = label.reshape(-1)
        return {'data': data, 'label': label}
