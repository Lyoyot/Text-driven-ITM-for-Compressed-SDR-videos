#!/usr/bin/env python
# -*- coding:utf-8 -*-
from torch.utils.data import DataLoader, DistributedSampler
import torch.utils.data as data
import h5py
import torch
import numpy as np
import os
import cv2
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from data.ICTCP_convert import SDR_to_ICTCP, HDR_to_ICTCP


class PNG_dataset(data.Dataset):
    def __init__(self, sdr_dir, gt_dir, cond_dir, num, is_random=True):
        super(PNG_dataset, self).__init__()
        self.sdr_list = []
        self.gt_list = []
        self.cond_list = []
        self._load_images_from_subfolders(sdr_dir, gt_dir, cond_dir)
        #name_list = os.listdir(sdr_dir)    # 遍历sdr_dir里面的每一个
        #for name in name_list:
            #self.sdr_list.append(sdr_dir + '/' + name)   # 将当前的文件名与输入路径连接，得到完整的文件路径
            #self.gt_list.append(gt_dir + '/' + name)
            #self.cond_list.append(cond_dir + '/' + name)
        if is_random:    # 检查是否需要进行随机打乱
            print(len(self.sdr_list))
            print(len(self.gt_list))
            print(len(self.cond_list))
            rnd_index = np.arange(len(self.sdr_list))    # 创建一个包含数据集索引的数组 rnd_index，其范围是从0到数据集长度减一
            print(rnd_index)
            np.random.shuffle(rnd_index)             # 使用 np.random.shuffle 函数对这个索引数组进行随机打乱
            self.sdr_list = np.array(self.sdr_list)[rnd_index]     # 使用打乱后的索引数组对 self.sdr_list 进行重新排序，以达到随机化的效果
            self.gt_list = np.array(self.gt_list)[rnd_index]
            self.cond_list = np.array(self.cond_list)[rnd_index]
        if num != 0:
            self.sdr_list = self.sdr_list[:num]
            self.gt_list = self.gt_list[:num]
            self.cond_list = self.cond_list[:num]

    def _load_images_from_subfolders(self, sdr_dir, gt_dir, cond_dir):
        # 获取子文件夹列表
        sdr_subfolders = [f.path for f in os.scandir(sdr_dir) if f.is_dir()]   # 用于获取指定目录下的所以子文件夹，并将这些子文件夹的路劲存储在‘subfolders’里面
        gt_subfolders = [f.path for f in os.scandir(gt_dir) if f.is_dir()]
        cond_subfolders = [f.path for f in os.scandir(cond_dir) if f.is_dir()]

        # 获取压缩表征cond（1x4096）
        for folder in cond_subfolders:
            folder_path = os.path.join(cond_dir, folder)
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    cond_path = os.path.join(root, file)
                    self.cond_list.append(cond_path)

        # 获取sdr图片（480x480）
        for folder in sdr_subfolders:
            folder_path = os.path.join(sdr_dir, folder)

            for subdir, _, files in os.walk(folder_path):
                # 选择固定下标范围的图片，按倒数第三个和倒数第二个下划线之间的数字进行排序
                selected_files = sorted(files, key=lambda x: int(x.split('_')[-3]))

                for image_file in selected_files[128:160]:  # 例如，选择排序后下标范围为10到19的图片
                    sdr_path = os.path.join(subdir, image_file)
                    self.sdr_list.append(sdr_path)
        # 获取ground-truth图片（480x480）
        for folder in gt_subfolders:
            folder_path = os.path.join(sdr_dir, folder)

            for subdir, _, files in os.walk(folder_path):
                # 选择固定下标范围的图片，按倒数第三个和倒数第二个下划线之间的数字进行排序
                selected_files = sorted(files, key=lambda x: int(x.split('_')[-3]))

                for image_file in selected_files[128:160]:  # 例如，选择排序后下标范围为10到19的图片
                    gt_path = os.path.join(subdir, image_file)
                    self.gt_list.append(gt_path)


    def __getitem__(self, index):
        #print(self.sdr_list)
        #print(self.gt_list)

        input_ = cv2.imread(self.sdr_list[index], flags=-1)[:,:,::-1]                #宽x高x通道数
        target_ = cv2.imread(self.gt_list[index], flags=-1)[:,:,::-1]                #cv.imread()读取通道的顺序三BGR，[:,:,::-1]转换为RGB

        input_ = np.array(input_, np.float32) / 255
        target_ = np.array(target_, np.float32) / 65535

        sdrRGB = torch.from_numpy(input_).float().permute(2, 0, 1)                   #通道数x宽x高
        gtRGB = torch.from_numpy(target_).float().permute(2, 0, 1)

        sdrITP = SDR_to_ICTCP(sdrRGB,dim=0)
        gtITP = HDR_to_ICTCP(gtRGB,dim=0)

        cond = np.load(self.cond_list[index])

        return {'sdrRGB': sdrRGB, 'gtRGB': gtRGB,
                'sdrITP': sdrITP, 'gtITP': gtITP,
                'cond': cond}

    def __len__(self):
        return len(self.sdr_list)


class Video_dataset(data.Dataset):
    def __init__(self, sdr_dir, gt_dir, num, is_random=True):
        super(Video_dataset, self).__init__()
        self.sdr_list = []
        self.gt_list = []
        name_list = os.listdir(sdr_dir)
        for name in name_list:
            self.sdr_list.append(sdr_dir + '/' + name)
            self.gt_list.append(gt_dir + '/' + name)
        if is_random:
            rnd_index = np.arange(len(self.sdr_list))
            np.random.shuffle(rnd_index)
            self.sdr_list = np.array(self.sdr_list)[rnd_index]
            self.gt_list = np.array(self.gt_list)[rnd_index]
        if num != 0:
            self.sdr_list = self.sdr_list[:num]
            self.gt_list = self.gt_list[:num]

    def __getitem__(self, index):
        #print(self.sdr_list)
        #print(self.gt_list)
        input_ = cv2.imread(self.sdr_list[index], flags=-1)[:,:,::-1]                #宽x高x通道数
        target_ = cv2.imread(self.gt_list[index], flags=-1)[:,:,::-1]                #cv.imread()读取通道的顺序三BGR，[:,:,::-1]转换为RGB

        input_ = np.array(input_, np.float32) / 255
        target_ = np.array(target_, np.float32) / 65535

        sdrRGB = torch.from_numpy(input_).float().permute(2, 0, 1)                   #通道数x宽x高
        gtRGB = torch.from_numpy(target_).float().permute(2, 0, 1)

        sdrITP = SDR_to_ICTCP(sdrRGB,dim=0)
        gtITP = HDR_to_ICTCP(gtRGB,dim=0)

        return {'sdrRGB': sdrRGB, 'gtRGB': gtRGB,
                'sdrITP': sdrITP, 'gtITP': gtITP}

    def __len__(self):
        return len(self.sdr_list)


class H5_dataset(data.Dataset):
    def __init__(self, file, num):
        super(H5_dataset, self).__init__()
        with h5py.File(file, 'r') as f:
            if num != 0:
                self.sdr = f['sdr'][:num]
                self.hdr = f['hdr'][:num]
            else:
                self.sdr = f['sdr'][()]
                self.hdr = f['hdr'][()]

    def __getitem__(self, index):
        sdr = self.sdr[index].astype('float32') / 255.0
        hdr = self.hdr[index].astype('float32') / 65535.0

        sdrRGB = torch.from_numpy(sdr).float().permute(2, 0, 1)
        gtRGB = torch.from_numpy(hdr).float().permute(2, 0, 1)

        sdrITP = SDR_to_ICTCP(sdrRGB, dim=0)
        gtITP = HDR_to_ICTCP(gtRGB, dim=0)

        return {'sdrRGB': sdrRGB, 'gtRGB': gtRGB,
                'sdrITP': sdrITP, 'gtITP': gtITP}

    def __len__(self):
        return len(self.sdr)


class Text_dataset(data.Dataset):
    def __init__(self, csv_path, num, is_random=True):
        super(Text_dataset, self).__init__()
        self.sdr_list = []
        self.gt_list = []
        self.text_list = []

        df = pd.read_csv(csv_path, sep="\t", header=None,  names=["filepath_sdr", "filepath_hdr", "text"], skiprows=1, encoding='utf-8')
        # df = pd.read_csv(csv_path, header=None, names=["filepath_sdr", "filepath_hdr", "text"], skiprows=1, encoding='utf-8')

        # # 读取每个键对应的值
        self.sdr_list = df['filepath_sdr'].tolist()
        self.gt_list = df['filepath_hdr'].tolist()
        self.text_list = df['text'].tolist()

        if is_random:    # 检查是否需要进行随机打乱
            rnd_index = np.arange(len(self.sdr_list))    # 创建一个包含数据集索引的数组 rnd_index，其范围是从0到数据集长度减一
            # print(rnd_index)
            np.random.shuffle(rnd_index)             # 使用 np.random.shuffle 函数对这个索引数组进行随机打乱
            self.sdr_list = np.array(self.sdr_list)[rnd_index]     # 使用打乱后的索引数组对 self.sdr_list 进行重新排序，以达到随机化的效果
            self.gt_list = np.array(self.gt_list)[rnd_index]
            self.text_list = np.array(self.text_list)[rnd_index]
        if num != 0:
            self.sdr_list = self.sdr_list[:num]
            self.gt_list = self.gt_list[:num]
            self.text_list = self.text_list[:num]

    def __getitem__(self, index):

        path = self.gt_list[index]

        input_ = cv2.imread(self.sdr_list[index], flags=-1)[:,:,::-1]                #宽x高x通道数
        # print(f"Trying to read image from: {path}")

        # target_ = cv2.imread(self.gt_list[index], flags=-1)[:,:,::-1]                #cv.imread()读取通道的顺序三BGR，[:,:,::-1]转换为RGB

        # 尝试加载图像
        target_ = cv2.imread(path, flags=-1)

        # 如果图像加载失败，则输出错误信息
        if target_ is None:
            print(f"Failed to load image from: {path}")

        target_ = target_[:,:,::-1]

        input_ = np.array(input_, np.float32) / 255
        target_ = np.array(target_, np.float32) / 65535

        sdrRGB = torch.from_numpy(input_).float().permute(2, 0, 1)    #通道数x宽x高
        gtRGB = torch.from_numpy(target_).float().permute(2, 0, 1)

        sdrITP = SDR_to_ICTCP(sdrRGB,dim=0)
        gtITP = HDR_to_ICTCP(gtRGB,dim=0)

        text = np.load(self.text_list[index])

        return {'sdrRGB': sdrRGB, 'gtRGB': gtRGB,
                'sdrITP': sdrITP, 'gtITP': gtITP,
                'text': text}
        # return {'sdrRGB': sdrRGB, 'text': text}

    def __len__(self):
        return len(self.sdr_list)



# class Text_dataset(data.Dataset):
#     def __init__(self, csv_path, num, is_random=True):
#         super(Text_dataset, self).__init__()
#         self.sdr_list = []
#         self.gt_list = []
#         self.text_list = []
#
#         df = pd.read_csv(csv_path, sep="\t", header=None,  names=["filepath_sdr", "filepath_hdr", 'compression_sdr', 'describe_sdr_content', 'describe_sdr_quality'], skiprows=1, encoding='utf-8')
#
#         # # 读取每个键对应的值
#         self.sdr_list = df['filepath_sdr'].tolist()
#         self.gt_list = df['filepath_hdr'].tolist()
#         self.compression_sdr = df['compression_sdr'].tolist()
#         self.describe_sdr_content = df['describe_sdr_content'].tolist()
#         self.describe_sdr_quality = df['describe_sdr_quality'].tolist()
#
#         if is_random:    # 检查是否需要进行随机打乱
#             rnd_index = np.arange(len(self.sdr_list))    # 创建一个包含数据集索引的数组 rnd_index，其范围是从0到数据集长度减一
#             # print(rnd_index)
#             np.random.shuffle(rnd_index)             # 使用 np.random.shuffle 函数对这个索引数组进行随机打乱
#             self.sdr_list = np.array(self.sdr_list)[rnd_index]     # 使用打乱后的索引数组对 self.sdr_list 进行重新排序，以达到随机化的效果
#             self.gt_list = np.array(self.gt_list)[rnd_index]
#             self.compression_list = np.array(self.compression_sdr)[rnd_index]
#             self.describe_sdr_content_list = np.array(self.compression_sdr)[rnd_index]
#             self.describe_sdr_quality_list = np.array(self.compression_sdr)[rnd_index]
#         if num != 0:
#             self.sdr_list = self.sdr_list[:num]
#             self.gt_list = self.gt_list[:num]
#             self.compression_list = self.compression_sdr[rnd_index]
#             self.describe_sdr_content_list = self.compression_sdr[rnd_index]
#             self.describe_sdr_quality_list = self.compression_sdr[rnd_index]
#
#     def __getitem__(self, index):
#
#         path = self.gt_list[index]
#
#         input_ = cv2.imread(self.sdr_list[index], flags=-1)[:,:,::-1]                #宽x高x通道数
#         # print(f"Trying to read image from: {path}")
#
#         # target_ = cv2.imread(self.gt_list[index], flags=-1)[:,:,::-1]                #cv.imread()读取通道的顺序三BGR，[:,:,::-1]转换为RGB
#
#         # 尝试加载图像
#         target_ = cv2.imread(path, flags=-1)
#
#         # 如果图像加载失败，则输出错误信息
#         if target_ is None:
#             print(f"Failed to load image from: {path}")
#
#         target_ = target_[:,:,::-1]
#
#         input_ = np.array(input_, np.float32) / 255
#         target_ = np.array(target_, np.float32) / 65535
#
#         sdrRGB = torch.from_numpy(input_).float().permute(2, 0, 1)    #通道数x宽x高
#         gtRGB = torch.from_numpy(target_).float().permute(2, 0, 1)
#
#         sdrITP = SDR_to_ICTCP(sdrRGB,dim=0)
#         gtITP = HDR_to_ICTCP(gtRGB,dim=0)
#
#         text = self.compression_list[index]
#         content = self.describe_sdr_content_list[index]
#         qulity = self.describe_sdr_quality_list[index]
#
#         return {'sdrRGB': sdrRGB, 'gtRGB': gtRGB,
#                 'sdrITP': sdrITP, 'gtITP': gtITP,
#                 'compression': text, 'content': content,
#                 'qulity':qulity}
#         # return {'sdrRGB': sdrRGB, 'text': text}
#
#     def __len__(self):
#         return len(self.sdr_list)
#


class Lcat_dataset(data.Dataset):
    def __init__(self, csv_path, num, is_random=True):
        super(Lcat_dataset, self).__init__()
        self.sdr_list = []
        self.gt_list = []
        self.cond_list = []

        df = pd.read_csv(csv_path, sep="\t", header=None,  names=["filepath_sdr", 'filepath_hdr', "title"], skiprows=1)

        # # 读取每个键对应的值
        self.sdr_list = df['filepath_sdr'].tolist()
        self.gt_list = df['filepath_hdr'].tolist()
        self.cond_list = df['title'].tolist()

        if is_random:    # 检查是否需要进行随机打乱
            rnd_index = np.arange(len(self.sdr_list))    # 创建一个包含数据集索引的数组 rnd_index，其范围是从0到数据集长度减一
            print(rnd_index)
            np.random.shuffle(rnd_index)             # 使用 np.random.shuffle 函数对这个索引数组进行随机打乱
            self.sdr_list = np.array(self.sdr_list)[rnd_index]     # 使用打乱后的索引数组对 self.sdr_list 进行重新排序，以达到随机化的效果
            self.gt_list = np.array(self.gt_list)[rnd_index]
            self.text_list = np.array(self.cond_list)[rnd_index]
        if num != 0:
            self.sdr_list = self.sdr_list[:num]
            self.gt_list = self.gt_list[:num]
            self.cond_list = self.cond_list[:num]

    def __getitem__(self, index):

        input_ = cv2.imread(self.sdr_list[index], flags=-1)[:,:,::-1]                #宽x高x通道数
        target_ = cv2.imread(self.gt_list[index], flags=-1)[:,:,::-1]                #cv.imread()读取通道的顺序三BGR，[:,:,::-1]转换为RGB

        input_ = np.array(input_, np.float32) / 255
        target_ = np.array(target_, np.float32) / 65535

        sdrRGB = torch.from_numpy(input_).float().permute(2, 0, 1)    #通道数x宽x高
        gtRGB = torch.from_numpy(target_).float().permute(2, 0, 1)

        sdrITP = SDR_to_ICTCP(sdrRGB,dim=0)
        gtITP = HDR_to_ICTCP(gtRGB,dim=0)

        text = self.cond_list[index]

        return {'sdrRGB': sdrRGB, 'gtRGB': gtRGB,
                'sdrITP': sdrITP, 'gtITP': gtITP,
                'text': text}

    def __len__(self):
        return len(self.sdr_list)




def create_dataset(opt, mode='train'):
    if mode == 'train':
        data_set = Text_dataset(opt.csv_train_path, opt.num)
    elif mode == 'val':
        data_set = Text_dataset(opt.csv_val_path, opt.num)
    data_loader = DataLoader(data_set, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                  shuffle=True)  # shuffle:在每个epoch开始的时候，对数据进行重新排序
    print('--PNG数据加载完成')

    return data_loader

class Text_dataset_blip(data.Dataset):
    def __init__(self, csv_path, num, is_random=True):
        super(Text_dataset_blip, self).__init__()
        self.sdr_list = []
        self.cond_list = []

        df = pd.read_csv(csv_path, sep="\t", header=None,  names=["filepath_sdr", "title"], skiprows=1)

        # # 读取每个键对应的值
        self.sdr_list = df['filepath_sdr'].tolist()
        self.cond_list = df['title'].tolist()
        if is_random:    # 检查是否需要进行随机打乱
            rnd_index = np.arange(len(self.sdr_list))    # 创建一个包含数据集索引的数组 rnd_index，其范围是从0到数据集长度减一
            print(rnd_index)
            np.random.shuffle(rnd_index)             # 使用 np.random.shuffle 函数对这个索引数组进行随机打乱
            self.sdr_list = np.array(self.sdr_list)[rnd_index]     # 使用打乱后的索引数组对 self.sdr_list 进行重新排序，以达到随机化的效果
            self.cond_list = np.array(self.cond_list)[rnd_index]
        if num != 0:
            self.sdr_list = self.sdr_list[:num]
            self.cond_list = self.cond_list[:num]

    def __getitem__(self, index):

        input_ = cv2.imread(self.sdr_list[index], flags=-1)[:,:,::-1]                #宽x高x通道数
        # input_ = Image.open(self.sdr_list[index]).convert("RGB")
        input_ = np.array(input_, np.float32)
        sdrRGB = torch.from_numpy(input_).float().permute(2, 0, 1)    #通道数x宽x高
        text = self.cond_list[index]

        return {'sdrRGB': sdrRGB, 'text': text}

    def __len__(self):
        return len(self.sdr_list)

def create_dataset_blip(opt):
    dataset = Text_dataset_blip(opt.csv_path_blip, opt.num)
    train_loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                              shuffle=True)  # shuffle:在每个epoch开始的时候，对数据进行重新排序
    print('--PNG数据加载完成')
    return train_loader



import random
def visualize_dataset(dataset, num_samples):
    # 随机选择 num_samples 个索引
    sample_indices = random.sample(range(len(dataset)), num_samples)

    for i, index in enumerate(sample_indices):
        data = dataset[index]
        sdr_img = data['sdrRGB'].permute(1, 2, 0).numpy().astype(np.uint8)
        gt_img = (data['gtRGB'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(sdr_img)
        plt.title(f'SDR Image\n{data["text"]}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(gt_img)
        plt.title('GT Image')
        plt.axis('off')

        plt.show()

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == '__main__':
    csv_path = r'I:\train_patch_ITP\daclip_train.csv'
    num_samples = 10  # 你想要可视化的样本数量
    dataset = Text_dataset(csv_path, num_samples)

    # 可视化前 10 个样本
    visualize_dataset(dataset, num_samples)

