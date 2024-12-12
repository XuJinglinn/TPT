import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from torchvision import transforms
from PIL import Image

class CustomCIFAR10Dataset(Dataset):
    def __init__(self, batch_file, transform=None):
        """
        自定义加载 CIFAR-10 数据集的类
        :param batch_file: CIFAR-10 处理后的 .pkl 文件路径
        :param transform: 数据预处理（如 CLIP 转换）
        """
        # 加载 .pkl 文件
        with open(batch_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        # 提取图像数据和标签
        self.images = data['data']
        self.labels = data['labels']
        
        # 变换
        self.transform = transform
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 读取图像数据并转换为 [32, 32, 3]
        img = self.images[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        
        # 如果定义了变换（如 CLIP_TRANSFORMS），应用变换
        if self.transform:
            img = Image.fromarray(img)  # 将 NumPy 数组转换为 PIL 图像对象
            img = self.transform(img)
        
        # 获取标签
        label = self.labels[idx]
        
        return img, label


def prepare_low_light_cifar_10_data(data_dir, degree=1, batch_size=128, num_workers=1):    
    batch_file = '/home/csh/Code/ETTA_Caus_VLM/WATT-New/data/Low_Light_CIFAR_10/test_batch_lowlight_'+ str(degree)
    dataset = CustomCIFAR10Dataset(batch_file=batch_file, transform=CLIP_TRANSFORMS)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    classes = dataset.classes
    print(f"low_light_cifar_10 dataset is selected, degree={str(degree)} , number of samples: {len(dataset)}!")
    return loader, classes
