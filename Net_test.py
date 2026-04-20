import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.backends.cudnn as cudnn
import os
import re
from tqdm import tqdm
from data.ICTCP_convert import SDR_to_ICTCP, HDR_to_ICTCP, ICTCP_to_HDR
from LCATNet.model_fusion_new import WholeNet_new
# from CSRNet_Dyn.model_csrnet import WholeNet
# import clip
# from lmdeploy.vl import load_image
from transformers import AutoTokenizer, AutoModelForCausalLM


class Net_test:
    def __init__(self, opt):
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # 第一个GPU
        self.load_dir = opt.load_dir
#        self.clip_load = opt.clip_load
        # self.model_tokenizer, self.preprocess = clip.load(self.clip_load, device=self.device)
        # self.llama = opt.llama_load
        # self.mllm = opt.mllm_load
        self.text_path = opt.text_path
        self.test_status()

    def test_status(self):
        # self.model_tokenizer, self.preprocess = clip.load(self.clip_load, device=self.device)
        # self.t_tokenizer = AutoTokenizer.from_pretrained(self.llama, use_fast=True, padding_side="left")

        self.model = WholeNet_new().to(self.device)
        # self.model = WholeNet(self.model_tokenizer, self.device).to(self.device)
        # self.model = WholeNet_new(self.model_tokenizer, self.device).to(self.device)

        checkpoint = torch.load(self.load_dir, map_location=self.device)
        self.model.CSRNet_I.load_state_dict(checkpoint['CSRNet_I'])
        self.model.CSRNet_TP.load_state_dict(checkpoint['CSRNet_TP'])
        self.model.fusionNet_new.load_state_dict(checkpoint['fusionNet_new'])
        print('--完成权重加载:{}--'.format(self.load_dir))
        if self.gpu_ids:
            self.model = nn.DataParallel(self.model, self.gpu_ids)  # 模型迁移到多个GPU上
        self.model.eval()

def compute_psnr(gt, hdr, peak=1.0):
    mse = np.mean(np.square(gt - hdr))
    psnr = 10 * np.log10(peak * peak / mse)
    return psnr

# def get_fifth_image(folder_path):
#     # 遍历文件夹
#     for root, dirs, files in os.walk(folder_path):
#         # 对文件排序
#         files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
#         # 获取第五张图片的路径
#         if len(files) >= 5:
#             fifth_image_path = os.path.join(root, files[4])
#             return fifth_image_path

def is_image_file(filename):
    # 定义图片文件的扩展名
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    # 获取文件扩展名并转换为小写
    ext = os.path.splitext(filename)[1].lower()
    return ext in image_extensions


def get_fifth_image(folder_path):
    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        # 过滤掉非图片文件
        image_files = [f for f in files if is_image_file(f)]
        # 对文件排序
        try:
            image_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        except ValueError as e:
            print(f"Error in sorting files: {e}")
            continue

        # 获取第五张图片的路径
        if len(image_files) >= 5:
            fifth_image_path = os.path.join(root, image_files[4])
            return fifth_image_path
        else:
            print("Not enough image files in the directory.")
    return None

def get_degra(filename):
    # 定义正则表达式模式
    pattern = re.compile(r'(H264_QP\d+|VP9_CRF\d+|H265_QP\d+|ASV2_QP\d+)')
    # 在文件名中搜索匹配的字符串
    match = pattern.search(filename)
    if match:
        filename = match.group(0)
        return filename
    return None

def transform_filename(source_filename):
    # 如果文件名包含'H264_QP'、'H265_QP'、'ASV2_QP'、'VP9_CRF'，进行替换并返回结果
    if any(x in source_filename for x in ['H264_QP', 'H265_QP', 'ASV2_QP', 'VP9_CRF']):
        # 替换'H264_QP'及其后的数字为'hdr'
        target_filename = re.sub(r'H264_QP\d+', 'hdr', source_filename)
        # 替换'H265_QP'及其后的数字为'hdr'
        target_filename = re.sub(r'H265_QP\d+', 'hdr', target_filename)
        # 替换'ASV2_QP'及其后的数字为'hdr'
        target_filename = re.sub(r'ASV2_QP\d+', 'hdr', target_filename)
        # 替换'VP9_CRF'及其后的数字为'hdr'
        target_filename = re.sub(r'VP9_CRF\d+', 'hdr', target_filename)
        return target_filename
    else:
        # 如果文件名包含'scenes<数字>'，在'scenes'之前加上'hdr_'
        if re.search(r'scenes\d+', source_filename):
            target_filename = re.sub(r'(scenes\d+)', r'hdr_\1', source_filename)
        else:
            # 如果没有'scenes'，在文件名最后加上'_hdr'
            target_filename = source_filename + '_hdr'
        return target_filename
#
# def describe_image_level(image_path, pipe, prompt='describe this image'):
#     image = load_image(image_path)
#     res = pipe((prompt, image))
#     return res.text
# # 假设文本特征是以.npy格式存储的
# def load_image(image_path):
#     image = cv2.imread(image_path, flags=-1)  # 读取图像
#     image = image[:, :, ::-1] / 255  # 转换为 RGB 并归一化
#     image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)  # 转换为 torch 张量
#     return image

def load_text_features(text_path):
    text_features = np.load(text_path)  # 加载文本特征
    text_features = torch.from_numpy(text_features).float()  # 转换为 torch 张量
    return text_features
from sklearn.decomposition import PCA
def apply_pca(features_file, target_dim=7680):
    # 加载特征数据
    features = np.load(features_file)
    print(f"原始特征形状: {features.shape}")

    # 如果特征的维度超过目标维度，则进行 PCA 降维
    if features.shape[1] > target_dim:
        pca = PCA(n_components=target_dim)
        reduced_features = pca.fit_transform(features)
        print(f"降维后的特征形状: {reduced_features.shape}")

        # 保存降维后的特征数据，新文件名为 _pca.npy 结尾
        output_file = features_file.replace(".npy", "_pca.npy")
        np.save(output_file, reduced_features)
        print(f"降维后的特征已保存到: {output_file}")
    else:
        print("特征维度小于或等于目标维度，无需降维。")


def test(opt):
    torch.manual_seed(901)  # 设置随机种子
    cudnn.benchmark = True  # 提高运行速度
    save_dir = opt.save_dir
    epoch = opt.epoch
    model = Net_test(opt)

    psnr_ITP_sum = []   # 初始化两个空列表，用于存储PSNR值
    psnr_RGB_sum = []
    list = os.listdir(opt.SDR_path)
    #list = list[0:2]
    for name in tqdm(list):

        sdr_file = get_fifth_image(os.path.join(opt.SDR_path, name))
        hdr_name = transform_filename(name)
        gt_file = get_fifth_image(os.path.join(opt.GT_path, hdr_name))
        file_name = os.path.basename(sdr_file)
        frame_name = file_name.split('.')[0]

        text_featrue_dir = os.path.join(opt.text_path, name)
        text_file = os.path.join(text_featrue_dir, frame_name, "summary_text_new_padded.npy")
        # text_file = os.path.join(text_featrue_dir, frame_name, "compression_sdr.npy")

        text_data = np.load(text_file)
        text_tensor = torch.from_numpy(text_data).float().to(opt.device)

        # sdr_file = opt.SDR_path + '/' + name
        # gt_file = opt.GT_path + '/' + name

        ERGB_sdr = cv2.imread(sdr_file, flags=-1)[:, :, ::-1] / 255
        ERGB_sdr = torch.from_numpy(ERGB_sdr).float().permute(2, 0, 1).unsqueeze(0).to(opt.device)

        ERGB_gt = cv2.imread(gt_file, flags=-1)[:, :, ::-1] / 65535
        EITP_gt = HDR_to_ICTCP(torch.from_numpy(ERGB_gt)).cpu().numpy()
        #name = name.split(".")[0] + '_i.png'
        with torch.no_grad():
            csr_result, fusion_result = model.model(ERGB_sdr, text_tensor)
            # csr_result, fusion_result = model.model(ERGB_sdr, compression_tensor, bright_tensor, color_tensor, content_tensor, detail_tensor)
            # csr_result = model.model(ERGB_sdr)
            # fusion_result = model.model(ERGB_sdr)
            ERGB_output1 = fusion_result.squeeze(0).permute(1, 2, 0).cpu().numpy()
            #ERGB_output2 = hdrRGB1.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if opt.save_img:
                # out_file = opt.OUT_path + '/' + name
                # os.makedirs(opt.OUT_path + '/' + name)
                # out_file = opt.OUT_path + '/' + name + '/' + os.path.basename(os.path.basename(sdr_file))
                # out_file = os.path.join(opt.OUT_path, name, os.path.basename(os.path.basename(sdr_file)))
                # out_file = concat_paths(sdr_file, opt.OUT_path)
                out_dir = os.path.join(opt.OUT_path, name)  # 只创建到目录级别
                os.makedirs(out_dir, exist_ok=True)  # 确保目录存在

                out_file = os.path.join(out_dir, os.path.basename(sdr_file))  # 生成完整文件路径
                # os.makedirs(out_file, exist_ok=True)  # 如果目录不存在，则创建
                cv2.imwrite(out_file, np.round(np.clip(ERGB_output1, a_min=0, a_max=1) * 65535)[:, :, ::-1].astype('uint16'))

            # 将 ERGB_output1 的值规范化到 [0, 1] 的范围内，同时将其转换为适合 16 位图像表示的值，然后再转换回 [0, 1] 范围内的浮点数。
            ERGB_hdr = np.round(np.clip(ERGB_output1, a_min=0, a_max=1) * 65535) / 65535
        EITP_hdr = HDR_to_ICTCP(torch.from_numpy(ERGB_hdr)).cpu().numpy()
        psnr_ITP = compute_psnr(EITP_gt, EITP_hdr)
        psnr_RGB = compute_psnr(ERGB_gt, ERGB_hdr)


        psnr_ITP_sum.append(psnr_ITP)
        psnr_RGB_sum.append(psnr_RGB)
    print()
    print(np.mean(psnr_ITP_sum))
    print(np.mean(psnr_RGB_sum))


    epoch_message = 'epoch:%d' \
                    'psnr_ITP:%.6f, psnr_RGB:%.6f'\
                    % (epoch,
                       np.mean(psnr_ITP_sum), np.mean(psnr_RGB_sum))
    # 确保目录存在
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    with open(save_dir, 'a', encoding='utf-8') as f:
        f.write(epoch_message)
        f.write('\n')






# def test(opt):
#     torch.manual_seed(901)
#     cudnn.benchmark = True
#     save_dir = opt.save_dir
#     epoch = opt.epoch
#     model = Net_test(opt)
#
#     psnr_ITP_sum = []
#     psnr_RGB_sum = []
#     list = os.listdir(opt.SDR_path)
#     #list = list[0:2]
#
#     # 遍历每一行并加载图像和文本特征
#     for index, row in df.iterrows():
#         image_path = row['image_path']
#         text_path = row['text_path']
#
#         # 加载图像和文本特征
#         image = load_image(image_path)
#         text_features = load_text_features(text_path)
#
#     for name in tqdm(list):
#         # sdr_file = opt.SDR_path + '/' + name
#         # gt_file = opt.GT_path + '/' + name
#
#         sdr_file = get_fifth_image(os.path.join(opt.SDR_path, name))
#         hdr_name = transform_filename(name)
#         gt_file = get_fifth_image(os.path.join(opt.GT_path, hdr_name))
#         # describe_sdr_content = describe_image_level(sdr_file, pipe, prompt='Describe the image.')
#         # describe_sdr_quality = describe_image_level(sdr_file, pipe, prompt='Rate the quality of the image.')
#         # cond_list = get_fifth_image(os.path.join(opt.cond, name))
#         #text = get_degra(name)
#
#         # compression_sdr_decoded = t_tokenizer(degradation_text, return_tensors='pt', add_special_tokens=False).to("cuda")
#         # describe_sdr_content_decoded = t_tokenizer(describe_sdr_content, return_tensors='pt',
#         #                                            add_special_tokens=False).to("cuda")
#         # describe_sdr_quality_decoded = t_tokenizer(describe_sdr_quality, return_tensors='pt',
#         #                                            add_special_tokens=False).to("cuda")
#
#         ERGB_sdr = cv2.imread(sdr_file, flags=-1)[:, :, ::-1] / 255
#         ERGB_sdr = torch.from_numpy(ERGB_sdr).float().permute(2, 0, 1).unsqueeze(0).to(opt.device)
#
#         ERGB_gt = cv2.imread(gt_file, flags=-1)[:, :, ::-1] / 65535
#         EITP_gt = HDR_to_ICTCP(torch.from_numpy(ERGB_gt)).cpu().numpy()
#
#         # text_input = clip.tokenize(degradation_text).to(opt.device)
#         # cond = np.array(cond_list)
#         # cond = np.load(cond_list)
#         # cond = torch.from_numpy(cond)
#
#         # text = np.load(cond_list)
#         #name = name.split(".")[0] + '_i.png'
#         with torch.no_grad():
#             # csr_result, fusion_result = model.model(ERGB_sdr, cond)
#
#             # compression_sdr_embedding = model(input_ids=compression_sdr_decoded['input_ids'], attention_mask=compression_sdr_decoded[
#             #                                       'attention_mask']).last_hidden_state
#             # compression_sdr_embeddings = compression_sdr_embedding.mean(dim=1)
#             # content_embedding = model(input_ids=describe_sdr_content_decoded['input_ids'],
#             #                           attention_mask=describe_sdr_content_decoded['attention_mask']).last_hidden_state
#             # content_embeddings = content_embedding.mean(dim=1)
#             # quality_embedding = model(input_ids=describe_sdr_quality_decoded['input_ids'],
#             #                           attention_mask=describe_sdr_quality_decoded['attention_mask']).last_hidden_state
#             # quality_embeddings = quality_embedding.mean(dim=1)
#
#             # csr_result = model.model(ERGB_sdr, compression_sdr_embeddings, content_embeddings, quality_embeddings)
#             # csr_result = model.model(ERGB_sdr)
#             # fusion_result = model.model(ERGB_sdr)
#             # ERGB_output1 = fusion_result[1].squeeze(0).permute(1, 2, 0).cpu().numpy()
#             ERGB_output1 = csr_result[1].squeeze(0).permute(1, 2, 0).cpu().numpy()
#             #ERGB_output2 = hdrRGB1.squeeze(0).permute(1, 2, 0).cpu().numpy()
#             if opt.save_img:
#                 # out_file = opt.OUT_path + '/' + name
#                 os.makedirs(opt.OUT_path + '/' + name)
#                 out_file = opt.OUT_path + '/' + name + '/' + os.path.basename(os.path.basename(sdr_file))
#                 cv2.imwrite(out_file,
#                             np.round(np.clip(ERGB_output1, a_min=0, a_max=1) * 65535)[:, :, ::-1].astype('uint16'))
#
#             ERGB_hdr = np.round(np.clip(ERGB_output1, a_min=0, a_max=1) * 65535) / 65535
#         EITP_hdr = HDR_to_ICTCP(torch.from_numpy(ERGB_hdr)).cpu().numpy()
#         psnr_ITP = compute_psnr(EITP_gt, EITP_hdr)
#         psnr_RGB = compute_psnr(ERGB_gt, ERGB_hdr)
#
#
#         psnr_ITP_sum.append(psnr_ITP)
#         psnr_RGB_sum.append(psnr_RGB)
#     print()
#     print(np.mean(psnr_ITP_sum))
#     print(np.mean(psnr_RGB_sum))
#
#
#     epoch_message = 'epoch:%d' \
#                     'psnr_ITP:%.6f, psnr_RGB:%.6f'\
#                     % (epoch,
#                        np.mean(psnr_ITP_sum), np.mean(psnr_RGB_sum))
#
#     with open(save_dir, 'a', encoding='utf-8') as f:
#         f.write(epoch_message)
#         f.write('\n')

