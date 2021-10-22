import os.path as osp
import json
import requests
import time
import numpy as np
import io
from PIL import Image
import logging
import torch
from torch.utils.data import Dataset

logger = logging.getLogger('global')


class ImageNetDataset(Dataset):
    """
    ImageNet Dataset.

    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - read_type (:obj:`str`): read type from the original meta_file
        - evaluator (:obj:`Evaluator`): evaluate to get metrics
        - image_reader_type (:obj:`str`): reader type 'pil' or 'ks'
        - server_cfg (list): server configurations

    Metafile example::
        "n01440764/n01440764_10026.JPEG 0\n"
    """
    def __init__(self, root_dir, meta_file, transform=None):
        super(ImageNetDataset, self).__init__()

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.transform = transform

        # read from local file
        with open(meta_file) as f:
            lines = f.readlines()

        self.num = len(lines)
        self.metas = []
        for line in lines:
            filename, label = line.rstrip().split()
            self.metas.append({'filename': filename, 'label': label})

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        curr_meta = self.metas[idx]
        filename = osp.join(self.root_dir, curr_meta['filename'])
        label = int(curr_meta['label'])
        # add root_dir to filename
        curr_meta['filename'] = filename
        with open(filename, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label
