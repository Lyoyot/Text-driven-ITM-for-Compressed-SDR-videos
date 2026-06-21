import os
import random
import sys

import numpy as np
import pandas as pd

import torch
from PIL import Image
from tqdm import tqdm
# from clip_interrogator import Interrogator, Config
# from clip_interrogator import Interrogator, Config
# from transformers import AutoModelForCausalLM, AutoConfig

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif']

# motion-blurry这个表示运动带来的模糊
# DEGRADATION_TYPES = ['motion-blurry','hazy','jpeg-compressed','low-light','noisy','raindrop','rainy','shadowed','snowy','uncompleted']
DEGRADATION_TYPES = ['avs_27', 'avs_32', 'avs_37', 'avs_42', 'H264_27', 'H264_32', 'H264_37', 'H264_42', 'H265_27', 'H265_32', 'H265_37', 'H265_42', 'vp9_27', 'vp9_32', 'vp9_37', 'vp9_42']
# DEGRADATION_TYPES = ['avs_27']

# DEGRADATION_TYPES = ['motion-blurry','H264-27','H264-32','H264-37','H264-42','avs-27','avs-32','avs-37','avs-42','H265-27','H265-32', 'H265-32', 'H265-37', 'H265-42']

# 定义了支持的图像文件扩展名列表
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_labels(folder_name, subfolders):
    labels = [0] * len(subfolders)
    labels[subfolders.index(folder_name)] = 1
    return labels

# 从文件夹中获取图像路径 函数会从给定的文件夹中递归地获取所有图像文件的路径，并返回一个路径列表
def _get_paths_from_images(path):
    '''get image path list from image folder'''
    # assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []   # 创建一个新列表
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        for dirpath, _, fnames in sorted(os.walk(folder_path)):
            selected_files = sorted(fnames, key=lambda x: int(x.split('_')[-3]))

            for image_file in selected_files[128:160]:  # 例如，选择排序后下标范围为10到19的图片
                image_path = os.path.join(dirpath, image_file)
                images.append(image_path)
                # 对选择的文件按照最后一个下划线和文件扩展名之间的数字进行排序
                selected_files = sorted(selected_files, key=lambda x: int(os.path.splitext(x.split('_')[-1])[0]))

    assert images, '{:s} has no valid image file'.format(path)
    return images


# 获取配对的图像路径和退化类型
# 函数会根据给定的数据根目录，获取低质量（LQ）和高质量（GT）图像对应的路径列表，并返回这些路径列表以及图像的退化类型（例如模糊、有雾、JPEG压缩等）
def get_paired_paths(dataroot):
    """
    Read LQ (Low Quality) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """
    GT_paths, LQ_paths, dagradations = [], [], []
    for deg_type in DEGRADATION_TYPES:  # 对每个退化类型进行循环
        paths1 = _get_paths_from_images(os.path.join(dataroot, deg_type, 'GT'))
        paths2 = _get_paths_from_images(os.path.join(dataroot, deg_type, 'LQ'))  # 然后每个退化路径里面都是gt和lq

        GT_paths.extend(paths1)  # GT list  # 添加到相应列表中
        LQ_paths.extend(paths2)  # LR list

        # 将退化类型的列表（deg_type）按照低质量图像路径列表（paths2）的长度进行扩展，以确保每个图像对都有对应的退化类型。
        dagradations.extend([deg_type] * len(paths2))
    print(f'GT length: {len(GT_paths)}, LQ length: {len(LQ_paths)}')
    return GT_paths, LQ_paths, dagradations


# 生成图像标注并保存为CSV文件
# 函数会根据给定的模式（'train'或'val'）生成图像的标注，并将图像路径和标注保存为CSV文件
def generate_captions(dataroot, ci, mode='train'):
    GT_paths, LQ_paths, dagradations = get_paired_paths(os.path.join(dataroot, mode))  # 获取图像路径和退化类型

    future_df = {"filepath": [], "title": []}  # 创建一个字典
    # 对于每一对高质量和低质量图像，它会使用 CLIP 模型（通过 Interrogator 类）生成图像的标注（caption）
    for gt_image_path, lq_image_path, dagradation in tqdm(zip(GT_paths, LQ_paths, dagradations)):
        image = Image.open(gt_image_path).convert('RGB')
        # caption = ci.generate_caption(image)
        # title = f'{caption}: {dagradation}'
        title = {dagradation}

        future_df["filepath"].append(lq_image_path)
        future_df["title"].append(title)

    pd.DataFrame.from_dict(future_df).to_csv(
        os.path.join(dataroot, f"daclip_{mode}.csv"), index=False, sep="\t"
    )

def get_paired_paths_text(dataroot):
    """
    Read LQ (Low Quality) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """
    LQ_paths, dagradations = [], []
    for deg_type in DEGRADATION_TYPES:  # 对每个退化类型进行循环
        paths = _get_paths_from_images(os.path.join(dataroot, deg_type))  # 然后每个退化路径里面都是gt和lq

        LQ_paths.extend(paths)  # LR list
        parts = deg_type.split("_")
        degradation_text = f"a photo is encoded by {parts[0]} with quality control parameter {parts[1]}"

        # 将退化类型的列表（deg_type）按照低质量图像路径列表（paths2）的长度进行扩展，以确保每个图像对都有对应的退化类型。
        dagradations.extend([degradation_text] * len(paths))
    return LQ_paths, dagradations

def generate_text(dataroot):
    LQ_paths, dagradations = get_paired_paths_text(os.path.join(dataroot))  # 获取图像路径和退化类型

    future_df = {"filepath": [], "title": []}  # 创建一个字典
    # 对于每一对高质量和低质量图像，它会使用 CLIP 模型（通过 Interrogator 类）生成图像的标注（caption）
    for lq_image_path, dagradation in tqdm(zip(LQ_paths, dagradations)):
        # image = Image.open(gt_image_path).convert('RGB')
        # caption = ci.generate_caption(image)
        # title = f'{caption}: {dagradation}'
        title = {dagradation}

        future_df["filepath"].append(lq_image_path)
        future_df["title"].append(title)

    pd.DataFrame.from_dict(future_df).to_csv(
        os.path.join(dataroot, f"daclip_text.csv"), index=False, sep="\t"
    )

if __name__ == "__main__":
    # dataroot = 'datasets/universal'
    dataroot = r'/mnt/hdd1/ljl/data/train_patch'

    # 创建了一个用于问答的 Interrogator 对象，并使用指定的 CLIP 模型作为其内部的视觉-文本编码器
    # 数据根目录 dataroot 和 CLIP 模型（通过 Interrogator 类）
    # ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
    # ci = Interrogator(Config(clip_model_name="ViT-L-14/openai", caption_model_name=None, clip_model_path=r"D:\pycharm-workspace\ITPNet_neew\pretrain_model\pytorch_model.bin"))  # 不指定模型名称

    # generate_captions(dataroot, ci, 'val')  # 为图像生成caption
    generate_text(dataroot)


