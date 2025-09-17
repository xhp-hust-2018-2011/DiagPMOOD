import ast
import io
import logging
import os

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Sampler

# from .base_dataset import BaseDataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class ImglistDataDataset():
    def __init__(self,
                 name,
                 imglist_pth,
                 data_dir,
                 preprocessor,
                 preprocessor_test,
                 num_classes=None,
                 **kwargs):
        # super(ImglistExtraDataDataset, self).__init__(**kwargs)

        self.name = name
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir
        # self.num_classes = num_classes
        # self.preprocessor = preprocessor
        self.transform_image = preprocessor
        self.transform_image_test = preprocessor_test
        # self.maxlen = maxlen
        self.total_num = len(self.imglist)

    def __len__(self):
        return self.total_num

    def __getitem__(self, index):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        path = os.path.join(self.data_dir, image_name)
        sample = dict()
        sample['image_name'] = image_name
        try:
            with open(path, 'rb') as f:
                content = f.read()
            filebytes = content
            buff = io.BytesIO(filebytes)
            image = Image.open(buff).convert('RGB')
            sample['data'] = self.transform_image(image)
            sample['data_test'] = self.transform_image_test(image)
            sample['label'] = int(extra_str)
        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        # return sample
        return sample




