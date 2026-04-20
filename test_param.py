import argparse
import torch
import os
import shutil
import re

def test_param(server, epoch, save_img):    # 训练参数？
    parser = argparse.ArgumentParser(description='')
    opt = parser.parse_args()
    opt.gpu_ids = [0]
    opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    opt.save_img = save_img
    opt.epoch = epoch
    if server:
        opt.SDR_path = r'/mnt/hdd1/ljl/data/choose_file_test'
        opt.GT_path = r'/mnt/hdd1/ljl/data/hdr_test'
        opt.OUT_path = 'LCATNET/results/model_{}'.format(epoch)          #save image
        opt.text_path = r'/mnt/hdd1/ljl/data/text_folder/test'

    else:
        # opt.SDR_path = r'Z:\ljl\choose_file_test'
        # opt.GT_path = r'Z:\ljl\HDR\whole\a_test'
        opt.SDR_path = r'D:\Lyy\datasets\KPNet\test\SDR'
        opt.GT_path = r'D:\Lyy\datasets\KPNet\test\hdr'
        # opt.text_path = r'Z:\ljl\owl_text_feature_test'
        # opt.text_path = r'D:\Lyy\Paper\CompressHDR\ljl\text_feature\test_feature\text'
        opt.text_path = r'D:\Lyy\datasets\KPNet\test\qp47'
        # opt.clip_load = r'D:\pycharm-workspace\ITPNet_neew\pretrain_model\ViT-B-32.pt'
        # opt.cond = r'E:\test_cond'
        # opt.OUT_path = 'LCATNET/results/model_{}'.format(epoch)          #save image
        opt.OUT_path = r'D:\Lyy\Paper\CompressHDR\ljl\ITPNet_neew\output\kpnet\model_{}'.format(epoch)
        # opt.clip_load = r'D:\pycharm-workspace\ITPNet_neew\pretrain_model\ViT-B-32.pt'
    if not os.path.exists(opt.OUT_path) and opt.save_img:
        os.makedirs(opt.OUT_path)

    # opt.load_dir = r'D:\pycharm-workspace\ITPNet_neew\CSRNet_Dyn\model_test\model_250_csr_youLSTM\model_148.pth'.format(epoch)
    # opt.load_dir = r'D:\pycharm-workspace\ITPNet_neew\LCATNet\model_alldata_model28_new_with_one_cross\model_{}.pth'.format(epoch)
    # opt.load_dir = r'/mnt/hdd1/ljl/code/ITPNet_neew/LCATNet/model_alldata_model28_new_with_one_cross/model_{}.pth'.format(epoch)
    # opt.clip_load = r'C:\Users\admin\.cache\huggingface\hub\models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K\snapshots\08f73555f1b2fb7c82058aebbd492887a94968ef\open_clip_pytorch_model.bin'
    # opt.clip_load = r'D:\pycharm-workspace\ITPNet_neew\pretrain_model\ViT-B-32.pt'
    opt.load_dir = r'D:\Lyy\Paper\CompressHDR\ljl\ITPNet_neew\LCATNet\models\all_data\model_{}.pth'.format(epoch)
    # opt.load_dir = r'FinetuneNet/model/model_{}.pth'.format(epoch)

    # opt.llama_load = r'D:\pycharm-workspace\pretrain_model\llama3.1-8b-instruct'
    # opt.mllm_load = r"D:\pycharm-workspace\pretrain_model\internlm-xcomposer2d5-7b"
    # opt.save_dir = opt.load_dir.rsplit("/", 1)[0] + '\psnr.txt'
    opt.save_dir = os.path.join(opt.OUT_path, '/psnr.txt')
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir, exist_ok=True)  # 创建目录
    print('Test start')

    from Net_test_recordtime import test
    # from test_csrnet import test
    test(opt)


if __name__ == '__main__':
    #for i in range(150, 151, 10):

    test_param(server=False, epoch=24, save_img=True)

