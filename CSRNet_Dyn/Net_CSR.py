from time import time
import kornia
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.cuda.amp as amp
# from tensorboardX import SummaryWriter
from torch.nn import init
import clip
import imageio
import cv2
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import sys

# sys.path.append('D:\pycharm-workspace\ITPNet_neew\datasets')
from data.dataset_ITP import create_dataset
# from datasets.dataset_ITP import create_dataset
from CSRNet_Dyn.model_csrnet import WholeNet
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import CLIPFeatureExtractor,CLIPVisionModel
from transformers import CLIPTokenizer
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from torchvision.utils import save_image
# # 定义文本模板
# template = "a photo are encoded by {} with {}"
# objects = ["cat", "dog", "car", "tree"]  # 你可以根据需要扩展这个列表

class Net:
    def __init__(self, opt):
        self.gpu_ids = opt.gpu_ids
        self.llama_load = opt.llama_load
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # 第一个GPU
        print(f"clip_load: {self.llama_load}")  # 调试信息
        if self.llama_load is None:
            raise ValueError("clip_load must be specified.")
        self.lr = opt.lr
        self.gamma = opt.gamma
        self.milestones = opt.milestones
        self.load_dir = opt.load_dir
        self.save_dir = opt.save_dir
        # self.save_image_dir = opt.save_image_dir
        self.tarin_status()
        self.Out_path = opt.save_image_dir

    # def tarin_status(self):
    #     print(f"Loading model from: {self.clip_load}")  # 调试信息
    #
    #     self.model_tokenizer, self.preprocess = clip.load(self.clip_load, device=self.device)
    #     # self.model_generation = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32)
    #     # self.preprocess_generation = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32)
    #     # 冻结文本编码器的参数
    #     self.model = WholeNet(self.model_tokenizer, self.device).to(self.device).cuda()
    #
    #     # self.model_tokenizer = CLIPTokenizer.from_pretrained(r'D:\pycharm-workspace\daclip-uir-main\pretrained\CLIP-ViT-B-32-laion2B-s34B-b79k').to(self.device)
    #     # self.preprocess = CLIPTextModel.from_pretrained(r'D:\pycharm-workspace\daclip-uir-main\pretrained\CLIP-ViT-B-32-laion2B-s34B-b79k').to(self.device)
    #
    #     self.set_loss_optimizer_scheduler()
    #     self.load_network()
    #     if self.gpu_ids:
    #         self.model = nn.DataParallel(self.model, self.gpu_ids)  # 模型迁移到多个GPU上
    #     self.model.train()

    def tarin_status(self):
        print(f"Loading model from: {self.llama_load}")  # 调试信息

        # self.model_tokenizer, self.preprocess = clip.load(self.llama_load, device=self.device)
        self.t_tokenizer = AutoTokenizer.from_pretrained(self.llama_load, use_fast=True, padding_side="left")
        # self.model_generation = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32)
        # self.preprocess_generation = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32)
        # 冻结文本编码器的参数
        self.model = WholeNet(self.t_tokenizer, self.device).to(self.device).cuda()

        # self.model_tokenizer = CLIPTokenizer.from_pretrained(r'D:\pycharm-workspace\daclip-uir-main\pretrained\CLIP-ViT-B-32-laion2B-s34B-b79k').to(self.device)
        # self.preprocess = CLIPTextModel.from_pretrained(r'D:\pycharm-workspace\daclip-uir-main\pretrained\CLIP-ViT-B-32-laion2B-s34B-b79k').to(self.device)

        self.set_loss_optimizer_scheduler()
        self.load_network()
        if self.gpu_ids:
            self.model = nn.DataParallel(self.model, self.gpu_ids)  # 模型迁移到多个GPU上
        self.model.train()

    def set_loss_optimizer_scheduler(self):
        self.L1 = nn.L1Loss().to(self.device)
        self.optim1 = optim.Adam(self.model.parameters(), lr=self.lr)     # 指定参数更新的网络  更新全局网络的参数
        self.sche1 = lr_scheduler.MultiStepLR(self.optim1, milestones=self.milestones, gamma=self.gamma)
        self.optimizers = [self.optim1]
        self.schedulers = [self.sche1]
        self.scaler = amp.GradScaler()


    def load_network(self):
         self.init_weight(self.model, 'xavier')
         if self.load_dir is not None:
             checkpoint = torch.load(self.load_dir, map_location=self.device)
             self.model.CSRNet_I.load_state_dict(checkpoint['CSRNet_I'])
             self.model.CSRNet_TP.load_state_dict(checkpoint['CSRNet_TP'])
             self.optim1.load_state_dict(checkpoint['Optimizer'])
             print('--完成权重加载:{}--'.format(self.load_dir))

    def init_weight(self, net, init_type):
            def init_func(m):
                classname = m.__class__.__name__
                if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                    if init_type == 'normal':
                        init.normal_(m.weight.data)
                    elif init_type == 'xavier':
                        init.xavier_normal_(m.weight.data)
                    elif init_type == 'kaiming':
                        init.kaiming_normal_(m.weight.data)
                    elif init_type == 'orthogonal':
                        init.orthogonal_(m.weight.data)
                    else:
                        raise NotImplementedError('initialization method {} is not implemented'.format(init_type))
                elif classname.find('BatchNorm2d') != -1:
                    init.normal_(m.weight.data)
                    init.constant_(m.bias.data, 0.0)

            print('--initialize network with {}'.format(init_type))
            net.apply(init_func)


    def get_current_lr(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups][0]

    def schedulers_step(self):
        for sche in self.schedulers:
            sche.step()

    def save_network(self, epoch):
        save_path = self.save_dir + '/8-12_model_{}.pth'.format(epoch)
        state = {
                 'CSRNet_I': self.model.module.CSRNet_I.state_dict(),
                 'CSRNet_TP': self.model.module.CSRNet_TP.state_dict(),
                 'Optimizer': self.optim1.state_dict(),
                 }
        torch.save(state, save_path)

    def train_step(self, data):
        for optim in self.optimizers:
            optim.zero_grad()

        """set data"""
        self.sdrRGB = data['sdrRGB'].to(self.device)
        self.sdrITP = data['sdrITP'].to(self.device)
        self.gtRGB = data['gtRGB'].to(self.device)
        self.gtITP = data['gtITP'].to(self.device)
        self.text = data['text'].to(self.device)
        # print(f"Input type: {self.text.dtype}")
        # 获取文本嵌入
        # self.gener_input = self.preprocess_generation(image=self.sdrRGB, return_tensors="pt")
        # generated_ids = self.preprocess_generation.generate(**self.gener_input)
        # generated_text = self.preprocess_generation.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # self.degra_text = clip.tokenize(self.degra).to(self.device)
        # self.text_input2 = clip.tokenize(generated_text).to(self.device)

        """cal loss"""
        self.hdrITP1, self.hdrRGB1 = self.model(self.sdrRGB, self.text)

        self.loss = self.L1(self.hdrITP1, self.gtITP)      #ITP空间loss
        # self.loss = self.L1(self.hdrRGB1, self.gtRGB)
        self.psnr_ITP = -1 * kornia.losses.psnr_loss(self.hdrITP1, self.gtITP, max_val=1)
        self.psnr_RGB = -1 * kornia.losses.psnr_loss(self.hdrRGB1, self.gtRGB, max_val=1)

        """back"""
        # self.loss.backward()
        self.scaler.scale(self.loss).backward()
        for optim in self.optimizers:
            self.scaler.step(optim)
        self.scaler.update()

    def crop_and_infer(self, image, crop_size=(960, 540)):
        h, w = image.shape[2], image.shape[3]
        crops = []
        for i in range(0, h, crop_size[0]):
            for j in range(0, w, crop_size[1]):
                cropped = image[:, :, i:i + crop_size[0], j:j + crop_size[1]]
                crops.append(cropped)
        # 进行推理
        results = []
        for crop in crops:
            results.append(self.model(crop, self.text_val))

        # 合并裁剪结果
        # 注意合并策略根据模型输出而定，以下仅是一个简单的示例
        output = torch.cat(results, dim=2)  # 需要根据实际情况调整
        return output

    def val_step(self, data):
        with torch.no_grad():  # 禁用梯度计算
            """set data"""
            self.sdrRGB_val = data['sdrRGB'].to(self.device)
            self.sdrITP_val = data['sdrITP'].to(self.device)
            self.gtRGB_val = data['gtRGB'].to(self.device)
            self.gtITP_val = data['gtITP'].to(self.device)
            self.text_val = data['text'].to(self.device)

            # self.degra_text_val = clip.tokenize(self.degra_val).to(self.device)

            """cal loss"""
            self.hdrITP1_val, self.hdrRGB1_val = self.model(self.sdrRGB_val, self.text_val)

            # 计算损失
            self.loss_val = self.L1(self.hdrITP1_val, self.gtITP_val)  # ITP 空间损失
            self.psnr_ITP_val = -1 * kornia.losses.psnr_loss(self.hdrITP1_val, self.gtITP_val, max_val=1)
            self.psnr_RGB_val = -1 * kornia.losses.psnr_loss(self.hdrRGB1_val, self.gtRGB_val, max_val=1)

            return self.loss.item(), self.psnr_ITP.item(), self.psnr_RGB.item(), self.hdrRGB1_val

    def tensorboard(self):
        loss = self.loss.item()
        psnr_ITP = self.psnr_ITP.item()
        psnr_RGB = self.psnr_RGB.item()

        return loss, psnr_ITP, psnr_RGB

def train(opt):
    torch.manual_seed(901)    # 设置随机种子
    train_loader = create_dataset(opt, mode='train')     # 创建了一个训练数据集的DataLoader
    # val_loader = create_dataset(opt, mode='val')
    batch_num = len(train_loader)       # 设置批次数量
    model = Net(opt)    # 配置模型
    writer = SummaryWriter(opt.save_dir, purge_step=0, filename_suffix=' ')
    psnr_val_max_RGB = 0
    epoch_best = 1

    for epoch in range(opt.epoch_start, opt.epoch_end + 1):   # 训练，迭代
        print("开始训练")
        losses_train_list = []
        psnres_train_ITP = []
        psnres_train_RGB = []    # 创建三个空列表，用于存储每个批次的损失、psnr值
        losses_val_list = []
        psnres_val_ITP = []
        psnres_val_RGB = []

        start = time()     # 记录当前时间，以便在训练结束后计算训练时间
        lr = model.get_current_lr()    # 获取当前的学习率
        # 训练阶段
        model.model.train()  # 切换到训练模式
        for i, data in enumerate(train_loader, 1): # 这是一个内部循环，用于遍历每个批次的训练数据
            model.train_step(data)

            loss,psnr_ITP,psnr_RGB = model.tensorboard()   # 调用模型的 tensorboard 方法，获取损失以及其他评估指标（这里是 PSNR）的值
            losses_train_list.append(loss)         # 将损失和评估指标的值添加到相应的列表中
            psnres_train_ITP.append(psnr_ITP)
            psnres_train_RGB.append(psnr_RGB)

            if i % 1000 == 0:    # 每十批次打印一次当前训练的一些信息，如学习率、损失和PSNR
                print('epoch:%d, batch:%d/%d, lr:%.7f,   '
                      'loss:%.6f, psnr_ITP1:%.2f, psnr_RGB1:%.2f, \n'
                      % (epoch, i, batch_num, lr, loss, psnr_ITP, psnr_RGB))
        loss_train_mean = np.mean(losses_train_list)
        psnr_train_mean_ITP = np.mean(psnres_train_ITP)
        psnr_train_mean_RGB = np.mean(psnres_train_RGB)

        model.model.eval()  # 确保模型处于评估模式
        # 在评估前释放内存
        '''
        with torch.no_grad():
            for i, data in enumerate(val_loader, 1):   # 这里的1代表着enumerate中的循环开始，默认为0
                val_loss, val_psnr_ITP, val_psnr_RGB, hdrRGB = model.val_step(data)
                losses_val_list.append(val_loss)
                psnres_val_ITP.append(val_psnr_ITP)
                psnres_val_RGB.append(val_psnr_RGB)
                epoch_folder_name = f"epoch_{epoch}"  # 根据当前epoch值生成文件夹名称
                epoch_dir = os.path.join(opt.save_image_dir, epoch_folder_name)
                os.makedirs(epoch_dir, exist_ok=True)
                if epoch % 20 == 0:
                    for j, image_tensor in enumerate(hdrRGB):
                        image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

                        # Save the validation images
                        save_path = f"{epoch_dir}/epoch_{epoch}_batch_{i}_image_{j}.png"
                        cv2.imwrite(save_path, np.round(np.clip(image, a_min=0, a_max=1) * 65535)[:, :, ::-1].astype('uint16'))
        
            # loss_val_mean = np.mean(losses_val_list)
            psnr_val_mean_ITP = np.mean(psnres_val_ITP)
            psnr_val_mean_RGB = np.mean(psnres_val_RGB)

        writer.add_scalars('loss', {'train': loss_train_mean, 'val': loss_val_mean}, global_step=epoch)
        writer.add_scalars('psnr_ITP', {'train': psnr_train_mean_ITP, 'val': psnr_val_mean_ITP}, global_step=epoch)
        writer.add_scalars('psnr_RGB', {'train': psnr_train_mean_RGB, 'val': psnr_val_mean_RGB}, global_step=epoch)

        model.schedulers_step()

        if psnr_val_mean_RGB > psnr_val_max_RGB:
            psnr_val_max_RGB = psnr_val_mean_RGB
            epoch_best = epoch
            model.save_network(epoch=epoch)
        '''
        writer.add_scalars('loss', {'train': loss_train_mean}, global_step=epoch)
        writer.add_scalars('psnr_ITP', {'train': psnr_train_mean_ITP}, global_step=epoch)
        writer.add_scalars('psnr_RGB', {'train': psnr_train_mean_RGB}, global_step=epoch)

        epoch_message = 'epoch:%d, batch_size:%d, lr:%.7f, time:%d, epoch_best:%d, loss:%.6f, psnr_YUV:%.2f, psnr_RGB:%.2f' \
                        % (epoch, opt.batch_size, lr, (time() - start) / 60, epoch_best,
                           loss_train_mean, psnr_train_mean_ITP,
                           psnr_train_mean_RGB)  # 打印周期数、批量大小、学习率、训练时间、平均损失以及平均PSNR

        #val_message = 'epoch:%d, batch_size:%d, val_loss:%.6f, val_psnr_YUV:%.2f, val_psnr_RGB:%.2f' \
        #             % (epoch, opt.batch_size, loss_val_mean, psnr_val_mean_ITP, psnr_val_mean_RGB)

        with open(opt.loss_file, 'a', encoding='utf-8') as f:     # 上述信息写入一个文件，这可能是用于记录训练过程中损失和评估指标的文件
            f.write(epoch_message)
            f.write('\n')
            #f.write(val_message)
            f.write('\n')
        print(epoch_message)
        #
        #print(val_message)
        print('------------')
        # model.schedulers_step()       # 用于更新学习率等调度器
        if epoch % opt.save_epoch == 0:    # 用于保存阶段性的模型参数
            model.save_network(epoch=epoch)

