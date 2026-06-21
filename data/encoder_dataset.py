
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 定义数据集类
class PairsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = self.generate_image_pairs()

    def generate_image_pairs(self):
        image_pairs = []
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if os.path.isdir(subdir_path):
                image_files = sorted(os.listdir(subdir_path))
                for i in range(len(image_files)):
                    for j in range(i+1, len(image_files)):
                        image_pairs.append((os.path.join(subdir_path, image_files[i]),
                                            os.path.join(subdir_path, image_files[j])))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image_path1, image_path2 = self.image_pairs[idx]
        image1 = Image.open(image_path1).convert("RGB")
        image2 = Image.open(image_path2).convert("RGB")

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2