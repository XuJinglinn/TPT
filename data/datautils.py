import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.hoi_dataset import BongardDataset
from data.fmow_dataset import FMoWDataset
from data.low_light_cifar10 import CustomCIFAR10Dataset
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations

ID_to_DIRNAME={
    'I': 'imagenet/images',
    'A': 'imagenet-adversarial',
    'K': 'ImageNet-Sketch',
    'R': 'imagenet-r',
    'V': 'imagenetv2-matched-frequency-format-val',
    'flower102': 'Flower102',
    'dtd': 'dtd',
    'pets': 'OxfordPets',
    'cars': 'StanfordCars',
    'ucf101': 'UCF101',
    'caltech101': 'Caltech101',
    'food101': 'Food101',
    'sun397': 'SUN397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat',
    # 'fmow': '',
}

def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id == 'FMoW':
        # "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
        testset = {}
        for year in range(0, 16):
            testset_item = FMoWDataset(env=year, mode=2, data_dir=data_root, transform=transform)
            # testset.append(testset_item)
            testset[year] = testset_item
            
    elif set_id == 'rmnist':
        testset = []
        for rotation_angle in range(0,90,10):
            testset_item = datasets.ImageFolder(root=data_root + '/rmnist/' + str(rotation_angle) + '/2', transform=transform)
            # testset.append(testset_item)
            testset[rotation_angle] = testset_item
            
    elif set_id == 'yearbook':
        testset = {}
        for year in range(1930, 2013, 4):
            testset_item = datasets.ImageFolder(root=data_root + '/yearbook/yearbook-merge/' + str(year) + '/2', transform=transform)
            # testset.append(testset_item)
            testset[year] = testset_item
            

    elif set_id == 'low_light_cifar10':
        testset = {}
        for degree in [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
            testset_item = CustomCIFAR10Dataset(batch_file=data_root + '/Low_Light_CIFAR_10/test_batch_lowlight_' + str(degree), transform=transform)
            # testset.append(testset_item)
            testset[degree] = testset_item
            
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(data_root, split, mode, base_transform, query_transform, bongard_anno)
    
        
    else:
        raise NotImplementedError
        
    return testset


# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views



