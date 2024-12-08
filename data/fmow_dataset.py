import os
import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset



class FMoWDataset(Dataset):
    def __init__(self, env=0, mode=2, data_dir='./data', transform=None):
        """
        Args:
            env (int): The environment index for selecting a specific subset of the dataset.
            mode (int): The mode index for selecting a specific subset of the dataset.
            data_dir (string): Directory with all the data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # self.datasets = pickle.load(open(os.path.join(data_dir, 'fmow.pkl'), 'rb'))
        self.data_dir = data_dir
        self.root = os.path.join(data_dir, 'fmow_v1.1')
        self.datasets = pickle.load(open(os.path.join(self.root, 'fmow.pkl'), 'rb'))
        
        self.env = env
        self.mode = mode
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the selected subset."""
        return len(self.datasets[self.env][self.mode]['labels'])

    def __getitem__(self, idx):
        """
        Args:
            idx (int or tensor): Index of the sample to be fetched.
        
        Returns:
            tuple: (image_tensor, label_tensor) where image_tensor is the transformed image and label_tensor is the label.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label_tensor = torch.LongTensor([self.datasets[self.env][self.mode]['labels'][idx]])
        image_tensor = self.transform(self.get_input(idx))
        return image_tensor, label_tensor

    def get_input(self, idx):
        """
        Args:
            idx (int): Index of the image to be fetched.
        
        Returns:
            PIL.Image: The image corresponding to the given index.
        """
        idx = self.datasets[self.env][self.mode]['image_idxs'][idx]
        img_path = os.path.join(self.root, 'images', f'rgb_img_{idx}.png')
        img = Image.open(img_path).convert('RGB')
        return img
