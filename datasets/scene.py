import os
import json
import math
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SceneDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split='train',
        data_fov=90,
        data_res=512,
        img_res=256,
        samples_per_epoch=10000,
        **kwargs
    ):
        self.samples_per_epoch = samples_per_epoch

        self.data_rgb = dict()
        self.data_depth = dict()
        self.data_camera = dict()

        self.data_dir = data_dir
        self.split = split
        self.datapath = os.path.join(self.data_dir, self.split)

        self.data_fov = data_fov
        self.data_res = data_res  # used 512 x 512 resolution for all test data
        self.img_res = img_res

        if 'mirror' in kwargs.keys():
            self.mirror = kwargs['mirror']
        else:
            self.mirror = False

        self.K = torch.Tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        self.K[0, 0] = self.K[0, 0] * (data_res * 0.5) / np.tan(np.deg2rad(data_fov) / 2)
        self.K[1, 1] = self.K[1, 1] * (data_res * 0.5) / np.tan(np.deg2rad(data_fov) / 2)

        downsampling_ratio = self.img_res / data_res
        self.K[0, 0] = self.K[0, 0] * downsampling_ratio
        self.K[1, 1] = self.K[1, 1] * downsampling_ratio

        seq_idxs = sorted([x for x in os.listdir(self.datapath) if not x.startswith('.')])
        self.images = []
        for idx in seq_idxs:
            if os.path.isdir(os.path.join(self.datapath, idx)):
                self.images += sorted([os.path.join(self.datapath, idx, x) for x in os.listdir(os.path.join(self.datapath, idx)) if x.endswith('.png')])
        self.images += [os.path.join(self.datapath, x) for x in seq_idxs if x.endswith('.png')]

        self.resize_transform_rgb = transforms.Compose([transforms.Resize(self.img_res), transforms.ToTensor()])

        self.rgb = []
        for img in self.images:
            self.rgb.append(self.resize_transform_rgb(Image.open(img))[:3, :, :])

        if self.samples_per_epoch <= 0:
            self.samples_per_epoch = len(self.rgb) * 2 if self.mirror else len(self.rgb)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        #random.seed()

        idxnum = idx % len(self.rgb)

        K = []
        rgb = []

        K.append(self.K)
        rgb.append(self.rgb[idxnum])

        K = torch.stack(K)
        rgb = torch.stack(rgb)

        if self.mirror and int(idx / len(self.rgb)) % 2 == 1:
            rgb = torch.flip(rgb, [-1])

        return {'rgb': rgb, 'K': K}

