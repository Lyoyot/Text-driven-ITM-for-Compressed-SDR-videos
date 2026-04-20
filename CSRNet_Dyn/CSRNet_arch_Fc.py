import functools
import math
import torch
import torch.nn as nn
from CSRNet_Dyn.convLSTM import ConvLSTMCell
import torch.nn.functional as F

class DynamicLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(DynamicLinear, self).__init__()
        self.in_features = in_features       # 输入特征的数量
        self.out_features = out_features     # 输出特征的数量
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))   # 定义权重参数‘weight’了，形状为（out, in），并将其注册为模型参数
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):   # 初始化权重和偏置
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))   #
        if self.bias is not None:      # 计算 fan_in 并根据 fan_in 的倒数平方根初始化偏置。
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, dynamic_weight):
        # dynamic_weight size: [batch_size, out_features, in_features]
        b, c = input.size()  # 获取输入的尺寸 (batch_size, in_features)
        input = input.view(1, -1, self.in_features)  # [1, batch_size * in_features]
        dynamic_weight = dynamic_weight.view(b * self.out_features, self.in_features)
        output = F.linear(input, dynamic_weight, self.bias)  # apply dynamic weight
        return output.view(b, self.out_features)

class HDA_FC(nn.Module):
    def __init__(self, in_features, fixed_out_features, dynamic_out_features, bias):
        super(HDA_FC, self).__init__()
        self.dynamic_fc = DynamicLinear(in_features, dynamic_out_features)
        self.fixed_fc = nn.Linear(in_features, fixed_out_features)
        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, dynamic_weight):
        out1 = self.dynamic_fc(x, dynamic_weight)
        out2 = self.fixed_fc(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        return out

class res_model(nn.Module):    # 一个块，里面是两个卷积和一个激活函数
    def __init__(self, chan_fixed, chan_dyn, num_feat, kernel_size, cond_nf, base_nf, in_nc):
        super(res_model, self).__init__()
        self.base_nf = base_nf
        self.cond_scale = HDA_FC(num_feat, chan_fixed, num_feat, bias=True)
        self.cond_shift = HDA_FC(num_feat, chan_fixed, num_feat, bias=True)
        self.conv = nn.Conv2d(base_nf, base_nf, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, cond_out, out_weight1, out_weight2):   # x:[4, 32, 480, 480]
        x = self.conv(x)
        scale = self.cond_scale(cond_out, out_weight1)    # [4, 64]
        shift = self.cond_shift(cond_out, out_weight2)    # [4, 64]
        out = x * scale.view(-1, self.base_nf, 1, 1) + shift.view(-1, self.base_nf, 1, 1) + x
        out = self.relu(out)
        return out

class Condition(nn.Module):
    def __init__(self, in_nc=3, nf=32):
        super(Condition, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, nf, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):             #
        conv1_out = self.act(self.conv1(self.pad(x)))  # [4, nf, 238,238]
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))   # [4, nf, 119, 119]
        conv3_out = self.act(self.conv3(self.pad(conv2_out)))     # [4, nf, 60, 60]
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)    # 全局池化GAP [4, nf]
        return out

class Condition_tp(nn.Module):     # [4, 32, 480, 480]
    def __init__(self, ino=64, nfo=32):
        super(Condition_tp, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(ino, nfo, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(nfo, nfo, 3, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nfo, nfo, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):             # [4, 32, 480, 480]
        conv1_out = self.act(self.conv1(self.pad(x)))  # [4, 32, 238, 238]
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))   # [4, 32, 119, 119]
        conv3_out = self.act(self.conv3(self.pad(conv2_out)))     # [4, 32, 60, 60]
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)    # 全局池化GAP [4, 32]
        return out   # [4, 32]

# 3layers with control
class CSRNet(nn.Module):    #
    def __init__(self, in_nc=3, out_nc=3, base_nf=64, cond_nf=32,
                 num_feat=64,
                 num_block=3,
                 res_scale=1,  # ？
                 img_range=255.,
                 kernel_size=3,
                 weight_ratio=0.5
                 ):
        super(CSRNet, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc

        self.cond_net = Condition(in_nc=in_nc, nf=cond_nf)

        self.img_range = img_range  # 图像范围
        self.num_feat = num_feat
        self.kernel_size = kernel_size  # 卷积核大小
        self.relu = nn.LeakyReLU(0.1, True)  # 激活函数
        self.num_block = num_block

        self.base_nf = base_nf
        self.out_nc = out_nc

        chan_fixed = int(num_feat * weight_ratio)  # 一半固定的参数   #
        chan_dyn = num_feat - chan_fixed  # 一半需要改变的参数

        self.conv_represent = nn.Sequential(  # 全连接层 DEM之后的那个全连接层
            nn.Linear(512, num_feat, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(num_feat, chan_dyn * num_feat * kernel_size * kernel_size, bias=False))

        self.forward_lstm = ConvLSTMCell((kernel_size, kernel_size), num_feat, num_feat, 1, bias=False)  # 前向
        # (kernel_size, kernel_size)表示输入张量的高度和宽度，num_feat表示输入张量的通道数，也表示hidden_dim隐藏状态的通道数。1表示卷积和大小。
        self.backward_lstm = ConvLSTMCell((kernel_size, kernel_size), num_feat, num_feat, 1, bias=False)  # 后向
        self.lstm_conv_list = nn.ModuleList()
        for _ in range(num_block * 2):  # 前后向之后的卷积  前项和后向各来一遍  就得是num_block*2
            self.lstm_conv_list.append(  # 将Conv2d添加进ModuleList列表中
                nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0)
            )

        self.conv_first = nn.Conv2d(in_nc, num_feat, 3, 1, 1)  # 第一个卷积 将输入通道数变为num_feat
        self.body = nn.ModuleList()
        for _ in range(num_block):
            self.body.append(
                res_model(  # 一个灰色（HDA）的输出
                    chan_fixed=chan_fixed,
                    chan_dyn=chan_dyn,
                    num_feat=num_feat,
                    kernel_size=kernel_size,
                    cond_nf=cond_nf,
                    base_nf=base_nf,
                    in_nc=cond_nf
                )
            )
        self.fc_layer = nn.Linear(64, 2048)

        self.cond_scale3 = nn.Linear(cond_nf, out_nc, bias=True)
        self.cond_shift3 = nn.Linear(cond_nf, out_nc, bias=True)
        self.conv3 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x, text):  # x[b,1,480,480]  text[8,512]
        cond = self.cond_net(x)      # 从条件网络得到一个输出 [b, cond_nf]

        represent = self.conv_represent(text)  # 最开始的全连接层
        represent = represent.view(-1, self.num_feat, self.kernel_size, self.kernel_size)  # [128, num_feat, 3, 3]

        h_init_for = torch.zeros_like(represent)  # 用于LSTM  用于初始化长短时记忆网络（LSTM）状态的张量
        c_init_for = torch.zeros_like(represent)
        next_state_for = (h_init_for, c_init_for)

        h_init_back = torch.zeros_like(represent)
        c_init_back = torch.zeros_like(represent)
        next_state_back = (h_init_back, c_init_back)  # [128, num_feat, 3, 3]

        out_weight_for = []
        out_weight_back = []

        for i in range(self.num_block * 2):  # 进行LSTM过程
            next_state_for = self.forward_lstm(represent, next_state_for)  # 前向
            out_weight_for.append(next_state_for[0])
            next_state_back = self.backward_lstm(represent, next_state_back)
            out_weight_back.append(next_state_back[0])

        out_weight_back.reverse()  # 是一个对列表或其他可迭代对象的反转操作

        x = self.conv_first(x)  # 卷积    # x: [b,num_feat,480,480]

        x1 = x.clone()

        for j in range(self.num_block):  # 残差快的块数
            # self.lstm_conv_list[j*2] 和 self.lstm_conv_list[j*2+1] 分别是两个 LSTM 单元
            out_weight1 = self.lstm_conv_list[j * 2](
                torch.cat((out_weight_for[j * 2], out_weight_back[j * 2]), dim=1))  # [128,64, 3,3] 将前向和后向方向的注意力权重拼接在一起
            out_weight2 = self.lstm_conv_list[j * 2 + 1](
                torch.cat((out_weight_for[j * 2 + 1], out_weight_back[j * 2 + 1]), dim=1))
            x1 = self.body[j](x1, cond, out_weight1, out_weight2)  # 用于残差块  out_wight1,2 [128,64,3,3]

        # 第三对FC
        scale3 = self.cond_scale3(cond)
        shift3 = self.cond_shift3(cond)
        # 这里使用view操作，将scale1和shift1的形状调整为(batch_size, self.base_nf, 1, 1)
        # 也就是相当于  out = out * scale1 + shift1 + out
        out = self.conv3(x1)
        out = out * scale3.view(-1, self.out_nc, 1, 1) + shift3.view(-1, self.out_nc, 1, 1) + out
        return out   # [4, 1, 480, 480]

class feature_mo(nn.Module):    # 最后只有一个卷积  没有双卷积
    def __init__(self, chan_fixed, chan_dyn, num_feat, kernel_size, cond_nf, out_nc, in_nc):
        super(feature_mo, self).__init__()
        self.cond_net = Condition_tp()
        self.out_nc = out_nc
        self.conv1 = nn.Conv2d(num_feat, out_nc, 1, 1, bias=True)
        self.cond_scale1 = nn.Linear(cond_nf, out_nc, bias=True)
        self.cond_shift1 = nn.Linear(cond_nf, out_nc, bias=True)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # [4, 64, 480, 480]
        cond_out = self.cond_net(x)   # [4, 32]
        scale1 = self.cond_scale1(cond_out)
        shift1 = self.cond_shift1(cond_out)   # [4, 2]

        out = self.conv1(x)   # [4, 2, 480, 480]  # 第一个卷积层
        out = out * scale1.view(-1, self.out_nc, 1, 1) + shift1.view(-1, self.out_nc, 1, 1) + out   #
        return out


class CSRNet_Dyn(nn.Module):
    def __init__(self, in_nc=3, out_nc=2, base_nf=64, cond_nf=32,
                 num_feat=64,
                 num_block=3,
                 res_scale=1,          # ？
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 kernel_size=3,
                 weight_ratio=0.5):
        super(CSRNet_Dyn, self).__init__()

        self.img_range = img_range  # 图像范围
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)  # 设置图像均值，用于将输入图像进行归一化。rgb_mean 是 RGB 图像的均值，被视为一个长度为 3 的元组
        self.num_feat = num_feat
        self.kernel_size = kernel_size  # 卷积核大小
        self.relu = nn.LeakyReLU(0.1, True)  # 激活函数
        self.num_block = num_block

        self.base_nf = base_nf
        self.out_nc = out_nc

        chan_fixed = int(num_feat * weight_ratio)  # 一半固定的参数   #
        chan_dyn = num_feat - chan_fixed  # 一半需要改变的参数

        self.conv_represent = nn.Sequential(  # 全连接层 DEM之后的那个全连接层
            nn.Linear(512, num_feat, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(num_feat, chan_dyn * num_feat * kernel_size * kernel_size, bias=False))

        self.forward_lstm = ConvLSTMCell((kernel_size, kernel_size), num_feat, num_feat, 1, bias=False)  # 前向
        # (kernel_size, kernel_size)表示输入张量的高度和宽度，num_feat表示输入张量的通道数，也表示hidden_dim隐藏状态的通道数。1表示卷积和大小。
        self.backward_lstm = ConvLSTMCell((kernel_size, kernel_size), num_feat, num_feat, 1, bias=False)  # 后向
        self.lstm_conv_list = nn.ModuleList()
        for _ in range(num_block * 2):  # 前后向之后的卷积  前项和后向各来一遍  就得是num_block*2
            self.lstm_conv_list.append(  # 将Conv2d添加进ModuleList列表中
                nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0)
            )

        self.conv_first = nn.Conv2d(in_nc, num_feat, 3, 1, 1)  # 第一个卷积 将输入通道数变为num_feat
        self.body = nn.ModuleList()
        for _ in range(num_block):
            self.body.append(
                )
        self.fc_layer = nn.Linear(64, 2048)

        self.feature_mo = feature_mo(
            chan_fixed=chan_fixed,
            chan_dyn=chan_dyn,
            num_feat=num_feat,
            kernel_size=kernel_size,
            out_nc=out_nc,
            cond_nf=cond_nf,
            in_nc=in_nc
        )

    def forward(self, x, cond):

        # 检查输入数据的类型
        print(f"Input type: {cond.dtype}")

        represent = self.conv_represent(cond)   # 最开始的全连接层
        represent = represent.view(-1, self.num_feat, self.kernel_size, self.kernel_size)  # [128, num_feat, 3, 3]

        h_init_for = torch.zeros_like(represent)  # 用于LSTM  用于初始化长短时记忆网络（LSTM）状态的张量
        c_init_for = torch.zeros_like(represent)
        next_state_for = (h_init_for, c_init_for)

        h_init_back = torch.zeros_like(represent)
        c_init_back = torch.zeros_like(represent)
        next_state_back = (h_init_back, c_init_back)   # [128, num_feat, 3, 3]

        out_weight_for = []
        out_weight_back = []

        for i in range(self.num_block * 2):  # 进行LSTM过程
            next_state_for = self.forward_lstm(represent, next_state_for)  # 前向
            out_weight_for.append(next_state_for[0])
            next_state_back = self.backward_lstm(represent, next_state_back)
            out_weight_back.append(next_state_back[0])

        # Replace LSTM outputs with zeros
        # out_weight_for = [torch.zeros_like(represent) for _ in range(self.num_block * 2)]
        # out_weight_back = [torch.zeros_like(represent) for _ in range(self.num_block * 2)]
        #
        out_weight_back.reverse()  # 是一个对列表或其他可迭代对象的反转操作

        x = self.conv_first(x)  # 卷积    # x: [4,con_fn,480,480]

        x1 = x.clone()

        for j in range(self.num_block):  # 残差快的块数
            # self.lstm_conv_list[j*2] 和 self.lstm_conv_list[j*2+1] 分别是两个 LSTM 单元
            out_weight1 = self.lstm_conv_list[j * 2](
                torch.cat((out_weight_for[j * 2], out_weight_back[j * 2]), dim=1))  # [128,64, 3,3] 将前向和后向方向的注意力权重拼接在一起
            out_weight2 = self.lstm_conv_list[j * 2 + 1](
                torch.cat((out_weight_for[j * 2 + 1], out_weight_back[j * 2 + 1]), dim=1))
            x1 = self.body[j](x1, out_weight1, out_weight2)     # 用于残差块  out_wight1,2 [128,64,3,3]

        x = self.feature_mo(x1)   # 输出得是[4, 2, 480, 480]
        return x
