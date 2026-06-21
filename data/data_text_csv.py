import os
import random
import sys
import re

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
DEGRADATION_TYPES = ['hazy']


# DEGRADATION_TYPES = ['motion-blurry','H264-27','H264-32','H264-37','H264-42','avs-27','avs-32','avs-37','avs-42','H265-27','H265-32', 'H265-32', 'H265-37', 'H265-42']

# 定义了支持的图像文件扩展名列表
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# 从文件夹中获取图像路径 函数会从给定的文件夹中递归地获取所有图像文件的路径，并返回一个路径列表
def _get_paths_from_images(path):
    '''get image path list from image folder'''
    # assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []   # 创建一个新列表

    for dirpath, _, fnames in sorted(os.walk(path)):
        selected_files = sorted(fnames, key=lambda x: int(x.split('_')[-3]))

        for image_file in selected_files[128:160]:  # 例如，选择排序后下标范围为10到19的图片
            image_path = os.path.join(dirpath, image_file)
            images.append(image_path)
            # 对选择的文件按照最后一个下划线和文件扩展名之间的数字进行排序
            selected_files = sorted(selected_files, key=lambda x: int(os.path.splitext(x.split('_')[-1])[0]))

    assert images, '{:s} has no valid image file'.format(path)
    return images


def get_degra(filename):
    # 定义正则表达式模式
    pattern = re.compile(r'(H264_QP\d+|VP9_CRF\d+|H265_QP\d+|ASV2_QP\d+)')
    # 在文件名中搜索匹配的字符串
    match = pattern.search(filename)
    if match:
        filename = match.group(0)
        return filename
    return None
        # print(match.group(0))  # 打印匹配的字符串

def transform_filename(source_filename):
    # 将'H264_QP'及其后的数字替换为'hdr'
    target_filename = re.sub(r'H264_QP\d+', 'hdr', source_filename)
    # 将'H265_QP'及其后的数字替换为'hdr'
    target_filename = re.sub(r'H265_QP\d+', 'hdr', target_filename)
    # 将'ASV2_QP'及其后的数字替换为'hdr'
    target_filename = re.sub(r'ASV2_QP\d+', 'hdr', target_filename)
    # 将'VP9_CRF'及其后的数字替换为'hdr'
    target_filename = re.sub(r'VP9_CRF\d+', 'hdr', target_filename)
    return target_filename

# 获取配对的图像路径和退化类型
# 函数会根据给定的数据根目录，获取低质量（LQ）和高质量（GT）图像对应的路径列表，并返回这些路径列表以及图像的退化类型（例如模糊、有雾、JPEG压缩等）
def get_paired_paths(dataroot_sdr, dataroot_hdr):
    """
    Read LQ (Low Quality) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """
    GT_paths, LQ_paths, dagradations = [], [], []

    # 获取当前文件夹下的所有子文件夹的名称
    sub_folders = [d for d in os.listdir(dataroot_sdr) if os.path.isdir(os.path.join(dataroot_sdr, d))]

    # 遍历每个子文件夹
    for sub_folder in sub_folders:

        files_hdr = transform_filename(sub_folder)
        degra = get_degra(sub_folder)
        if degra is None:
            print(f"Skipping folder '{sub_folder}' as it does not match the expected pattern.")
            continue
        degra = degra.split("_")
        degradation_text = f"a photo is encoded by {degra[0]} with quality control parameter {degra[1][-2:]}"

        paths1 = _get_paths_from_images(os.path.join(dataroot_sdr, sub_folder))
        paths2 = _get_paths_from_images(os.path.join(dataroot_hdr, files_hdr))

        GT_paths.extend(paths2)  # GT list  # 添加到相应列表中
        LQ_paths.extend(paths1)  # LR list
        dagradations.extend([degradation_text] * len(paths1))
        # dagradations.append(degradation_text)


    print(f'GT length: {len(GT_paths)}, LQ length: {len(LQ_paths)}')
    return GT_paths, LQ_paths, dagradations

import csv

# 生成图像标注并保存为CSV文件
# 函数会根据给定的模式（'train'或'val'）生成图像的标注，并将图像路径和标注保存为CSV文件
def generate_captions(dataroot_sdr, dataroot_hdr, mode='train'):
    # GT_paths, LQ_paths, dagradations = get_paired_paths(os.path.join(dataroot_sdr, mode), dataroot_hdr)  # 获取图像路径和退化类型
    GT_paths, LQ_paths, dagradations = get_paired_paths(dataroot_sdr, dataroot_hdr)  # 获取图像路径和退化类型

    future_df = {"filepath_sdr": [], "filepath_hdr": [], "title": []}  # 创建一个字典
    # 对于每一对高质量和低质量图像，它会使用 CLIP 模型（通过 Interrogator 类）生成图像的标注（caption）
    # image = Image.open(gt_image_path).convert('RGB')
    # caption = ci.generate_caption(image)
    # title = f'{dagradations}'

    for gt_image_path, lq_image_path, dagradation in tqdm(zip(GT_paths, LQ_paths, dagradations)):

        future_df["filepath_sdr"].append(lq_image_path)
        future_df["filepath_hdr"].append(gt_image_path)
        future_df["title"].append(dagradation)

    pd.DataFrame.from_dict(future_df).to_csv(
        os.path.join(dataroot_sdr, f"daclip_val_250.csv"), index=False, sep="\t"
    )

    # future_df["filepath_sdr"].extend(LQ_paths)
    # future_df["filepath_hdr"].extend(GT_paths)
    # future_df["title"].extend(dagradations)
    #
    # # 创建DataFrame对象
    # df = pd.DataFrame(future_df)
    #
    # # 将数据写入CSV文件，每个字段使用制表符分隔
    # df.to_csv(
    #     os.path.join(dataroot_sdr, f"daclip_{mode}.csv"), index=False, sep="\t", header=False
    # )


if __name__ == "__main__":
    # dataroot = 'datasets/universal'
    # dataroot_sdr = r'I:\train_patch_ITP\500'
    # dataroot_hdr = r'I:\hdr_train_patch\500'
    # dataroot_sdr = r'I:\test_itp\sdr'
    # dataroot_hdr = r'I:\test_itp\hdr'
    dataroot_sdr = r'F:\val_patch'
    dataroot_hdr = r'F:\hdr_val_patch'

    # dataroot_sdr = r'/mnt/hdd1/ljl/data/train_patch_itp/250'
    # dataroot_hdr = r'/mnt/hdd1/ljl/data/hdr_patch/250'


    # 创建了一个用于问答的 Interrogator 对象，并使用指定的 CLIP 模型作为其内部的视觉-文本编码器
    # 数据根目录 dataroot 和 CLIP 模型（通过 Interrogator 类）
    # ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

    # generate_captions(dataroot_sdr, dataroot_hdr, 'val')  # 为图像生成caption
    generate_captions(dataroot_sdr, dataroot_hdr, 'train')


