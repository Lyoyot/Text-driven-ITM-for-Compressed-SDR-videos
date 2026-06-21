import os
import re
import pandas as pd
from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif']

# 定义了支持的图像文件扩展名列表
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def extract_number_from_filename(file_name):
    # 从文件名中提取最后一个下划线后的数字字符
    match = re.search(r'_(\d+)', file_name)
    if match:
        return int(match.group(1))
    return 0

# 从文件夹中获取图像路径 函数会从给定的文件夹中递归地获取所有图像文件的路径，并返回一个路径列表
def _get_paths_from_images(path):
    '''get image path list from image folder'''
    # assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    filenames = os.listdir(path)
    filenames = [f for f in filenames if os.path.isfile(os.path.join(path, f))]
    # 按照文件名中的数字部分排序
    selected_files = sorted(filenames, key=lambda x: int(os.path.splitext(x.split('_')[1])[0]))

    # 检查排序后的列表是否有至少5个元素
    if len(selected_files) >= 5:
        # 选取第五张图片
        image_file = selected_files[4]
        image_path = os.path.join(path, image_file)
    # for dirpath, _, frames in sorted(os.walk(path)):
    #     # 按照文件名中的数字部分排序
    #     selected_files = sorted(frames, key=lambda x: int(os.path.splitext(x.split('_')[1])[0]))
    #
    #     # 检查排序后的列表是否有至少5个元素
    #     if len(selected_files) >= 5:
    #         # 选取第五张图片
    #         image_file = selected_files[4]
    #         image_path = os.path.join(dirpath, image_file)

    assert image_path, '{:s} has no valid image file'.format(path)
    return image_path

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

        GT_paths.append(paths2)  # GT list  # 添加到相应列表中
        LQ_paths.append(paths1)  # LR list
        dagradations.extend([degradation_text] * len(paths1))
        # dagradations.append(degradation_text)


    print(f'GT length: {len(GT_paths)}, LQ length: {len(LQ_paths)}')
    return GT_paths, LQ_paths, dagradations

# 生成图像标注并保存为CSV文件
# 函数会根据给定的模式（'train'或'val'）生成图像的标注，并将图像路径和标注保存为CSV文件
def generate_captions(dataroot_sdr, dataroot_hdr, mode='train'):
    # GT_paths, LQ_paths, dagradations = get_paired_paths(os.path.join(dataroot_sdr, mode), dataroot_hdr)  # 获取图像路径和退化类型
    GT_paths, LQ_paths, dagradations = get_paired_paths(dataroot_sdr, dataroot_hdr)  # 获取图像路径和退化类型

    future_df = {"filepath_sdr": [], "filepath_hdr": [], "title": []}  # 创建一个字典

    for gt_image_path, lq_image_path, dagradation in tqdm(zip(GT_paths, LQ_paths, dagradations)):

        future_df["filepath_sdr"].append(lq_image_path)
        future_df["filepath_hdr"].append(gt_image_path)
        future_df["title"].append(dagradation)

    pd.DataFrame.from_dict(future_df).to_csv(
        os.path.join(dataroot_sdr, f"daclip_val.csv"), index=False, sep="\t"
    )


if __name__ == "__main__":
    # dataroot = 'datasets/universal'
    # dataroot_sdr = r'I:\train_patch_ITP'
    # dataroot_hdr = r'I:\hdr_train_patch'
    # dataroot_sdr = r'E:\val'
    # dataroot_hdr = r'G:\HDR\whole\a_val'
    # dataroot_sdr = r'/mnt/hdd1/ljl/data/val'
    # dataroot_hdr = r'/mnt/hdd1/ljl/data/hdr_val'

    dataroot_sdr = r'E:\test'
    dataroot_hdr = r'G:\HDR\whole\a_test'


    generate_captions(dataroot_sdr, dataroot_hdr, 'train')


