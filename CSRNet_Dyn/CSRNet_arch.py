import functools
import math
import torch
import torch.nn as nn
from CSRNet_Dyn.convLSTM import ConvLSTMCell
import torch.nn.functional as F

class conv_dyn(nn.Conv2d):   # 如何对需要生成的参数  进行卷积

    def _conv_forward(self, input, weight, bias, groups):

        if self.padding_mode != 'zeros':
            # F.pad(input, pad, mode='constant', value=0) 对tensor进行扩充  input:需要扩充的tensor，pad：扩充维度，mode：扩充方法 value：扩充时指定补充值
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride, (0, 0), self.dilation, groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, groups)

    def forward(self, input, weight):
        # weight size: [b*out, in ,k, k]
        b, c, h, w = input.size()    # 获取输入张量的尺寸信息
        conv_weight = self.weight.repeat(b, 1,1,1)   # 【124，64，3】 repeat()给三个维度各加一行  将网络的卷积权重进行扩展
        # out = self._conv_forward(input.view(1, -1, h, w), conv_weight * weight, self.bias, groups=b)    # 最后的输出结果
        out = self._conv_forward(input.view(1, -1, h, w), conv_weight * weight, self.bias, groups=b)
        return out.view(b, -1, h, w)  # 将一维变为b维

class HDA_conv(nn.Module):   # SDL
    def __init__(self, chan_fixed, chan_dyn, num_feat, kernel_size):
        super(HDA_conv, self).__init__()

        self.kernel_size = kernel_size  # 卷积核大小
        self.num_feat = num_feat

        self.conv_share = nn.Conv2d(num_feat, chan_fixed, kernel_size, stride=1, padding=(kernel_size-1)//2)
        # 需要修改的参数部分
        self.conv_dyn = conv_dyn(num_feat, chan_dyn, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, out_weight):
        # calculate conv weights
        # out_weight size: [chan_dyn, num_feat, kernel_size, kernel_size]    out_wight：[32, 64, 3, 3]

        # b, c, h, w = x.size()

        out1 = self.conv_dyn(x, out_weight)  # [4, 128, 480, 480]
        out2 = self.conv_share(x)

        out = torch.cat((out1, out2), dim=1)  # dim维度 在横向叠加  SDL

        return out

class DynamicLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(DynamicLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, dynamic_weight):
        # dynamic_weight size: [batch_size, out_features, in_features]
        batch_size = input.size(0)
        input = input.view(1, -1, self.in_features)  # [1, batch_size * in_features]
        dynamic_weight = dynamic_weight.view(batch_size * self.out_features, self.in_features)
        output = F.linear(input, dynamic_weight, self.bias)  # apply dynamic weight
        return output.view(batch_size, self.out_features)

class HDA_FC(nn.Module):
    def __init__(self, in_features, fixed_out_features, dynamic_out_features):
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
    def __init__(self, in_nc=3, out_nc=3, base_nf=64, cond_nf=32):
        super(CSRNet, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc

        self.cond_net = Condition(in_nc=in_nc, nf=cond_nf)

        self.cond_scale1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.conv1 = nn.Conv2d(in_nc, base_nf, 1, 1, bias=True)

        self.cond_scale2 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift2 = nn.Linear(cond_nf, base_nf, bias=True)
        self.conv2 = nn.Conv2d(base_nf, base_nf, 1, 1, bias=True)

        self.cond_scale3 = nn.Linear(cond_nf, out_nc, bias=True)
        self.cond_shift3 = nn.Linear(cond_nf, out_nc, bias=True)
        self.conv3 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        cond = self.cond_net(x)      # 从条件网络得到一个输出 [4, nf]

        # 第一对FC
        scale1 = self.cond_scale1(cond)     # [4, base_nf]  Linear层  线性层，用于学习输入特征与输出特征之间的线性映射  conf_nf:32-->base_nf:64
        shift1 = self.cond_shift1(cond)     # [4, base_nf]也是线性层

        # 第二对FC
        scale2 = self.cond_scale2(cond)
        shift2 = self.cond_shift2(cond)

        # 第三对FC
        scale3 = self.cond_scale3(cond)
        shift3 = self.cond_shift3(cond)

        out = self.conv1(x)    # 第一个卷积层
        # 这里使用view操作，将scale1和shift1的形状调整为(batch_size, self.base_nf, 1, 1)
        # 也就是相当于  out = out * scale1 + shift1 + out
        out = out * scale1.view(-1, self.base_nf, 1, 1) + shift1.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out)      # 激活层

        out = self.conv2(out)
        out = out * scale2.view(-1, self.base_nf, 1, 1) + shift2.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out)      # 激活层

        out = self.conv3(out)
        out = out * scale3.view(-1, self.out_nc, 1, 1) + shift3.view(-1, self.out_nc, 1, 1) + out
        return out   # [4, 1, 480, 480]

class res_model(nn.Module):    # 一个块，里面是两个卷积和一个激活函数
    def __init__(self, chan_fixed, chan_dyn, num_feat, kernel_size, cond_nf, base_nf, res_scale=1.0):
        super(res_model, self).__init__()
        self.res_scale = res_scale
        self.cond_net = Condition_tp()
        self.base_nf = base_nf
        self.conv1 = HDA_conv(chan_fixed, chan_dyn, num_feat, kernel_size)
        self.cond_scale = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift = nn.Linear(cond_nf, base_nf, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = HDA_conv(chan_fixed, chan_dyn, num_feat, kernel_size)

    def forward(self, x, out_weight1, out_weight2):   # x:[4, 32, 480, 480]
        id = x
        cond_out = self.cond_net(x)   # cond_out [4, 32]
        scale = self.cond_scale(cond_out)    # [4, 64]
        shift = self.cond_shift(cond_out)    # [4, 64]

        out = self.conv1(x, out_weight1)  # 第一个卷积层  x:[4,2,480,480]   out_weight1: [128, 64, 3, 3]  out:[4, 128, 480,480]
        out = out * scale.view(-1, self.base_nf, 1, 1) + shift.view(-1, self.base_nf, 1, 1) + out
        out = self.relu(out)
        out = self.conv2(out, out_weight2)
        out = id + out * self.res_scale  # 新加上的

        return out

class feature_mo(nn.Module):    # 最后只有一个卷积  没有双卷积
    def __init__(self, num_feat, cond_nf, out_nc):
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

class AdaptiveLinear(nn.Module):
    def __init__(self, output_dim):
        super(AdaptiveLinear, self).__init__()
        self.output_dim = output_dim
        self.linear = None  # 延迟初始化

    def forward(self, x):
        input_dim = x.size(-1)
        if self.linear is None or self.linear.in_features != input_dim:
            self.linear = nn.Linear(input_dim, self.output_dim).to(x.device)
        return self.linear(x)


class CSRNet_Dyn(nn.Module):
    def __init__(self, in_nc=3, out_nc=2, base_nf=64, cond_nf=32,
                 num_feat=64,
                 num_block=3,
                 res_scale=1,          # ？
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 kernel_size=1,
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
            nn.Linear(7680, num_feat, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(num_feat, chan_dyn * num_feat * kernel_size * kernel_size, bias=False))

        # self.conv_represent = nn.Sequential(  # 全连接层 DEM之后的那个全连接层
        #     AdaptiveLinear(num_feat),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(num_feat, chan_dyn * num_feat * kernel_size * kernel_size, bias=False))

        self.forward_lstm = ConvLSTMCell((kernel_size, kernel_size), num_feat, num_feat, 1, bias=False)  # 前向
        # (kernel_size, kernel_size)表示输入张量的高度和宽度，num_feat表示输入张量的通道数，也表示hidden_dim隐藏状态的通道数。1表示卷积和大小。
        self.backward_lstm = ConvLSTMCell((kernel_size, kernel_size), num_feat, num_feat, 1, bias=False)  # 后向
        self.lstm_conv_list = nn.ModuleList()
        for _ in range(num_block * 2):  # 前后向之后的卷积  前项和后向各来一遍  就得是num_block*2
            self.lstm_conv_list.append(  # 将Conv2d添加进ModuleList列表中
                nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0)
            )

        self.conv_first = nn.Conv2d(in_nc, num_feat, 1, 1, 0)  # 第一个卷积 将输入通道数变为num_feat
        self.body = nn.ModuleList()
        for _ in range(num_block):
            self.body.append(
                res_model(  # 一个灰色（HDA）的输出
                    chan_fixed=chan_fixed,
                    chan_dyn=chan_dyn,
                    num_feat=num_feat,
                    cond_nf=cond_nf,
                    kernel_size=kernel_size,
                    base_nf=base_nf
                ))
        self.fc_layer = nn.Linear(64, 2048)

        self.feature_mo = feature_mo(
            num_feat=num_feat,
            out_nc=out_nc,
            cond_nf=cond_nf
        )

    def forward(self, x, cond):

        # 检查输入数据的类型
        # print(f"Input type: {cond.dtype}")
        # type = cond.shape
        # print(type)

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


    # def forward(self, x, cond):
    #     # print('x', x.shape)     # [4,2,480,480]
    #     # cond_out = self.cond_net(x)      # 从条件网络得到一个输出
    #     # print('cond_out', cond_out.shape)    # [4, 32]
    #     # cond_out = cond_out.view(4, 32)
    #
    #     # represent = cond.view(cond.size(0), -1)  # 将输入展平为 [batch_size, 4096]  [4, 4096]
    #     represent = self.conv_represent(cond)
    #     represent = represent.view(-1, self.num_feat, self.kernel_size, self.kernel_size)  # [128, num_feat, 3, 3]
    #
    #     h_init_for = torch.zeros_like(represent)  # 用于LSTM  用于初始化长短时记忆网络（LSTM）状态的张量
    #     c_init_for = torch.zeros_like(represent)
    #     next_state_for = (h_init_for, c_init_for)
    #
    #     h_init_back = torch.zeros_like(represent)
    #     c_init_back = torch.zeros_like(represent)
    #     next_state_back = (h_init_back, c_init_back)   # [128, num_feat, 3, 3]
    #
    #     out_weight_for = []
    #     out_weight_back = []
    #
    #     for i in range(self.num_block * 2):  # 进行LSTM过程
    #         next_state_for = self.forward_lstm(represent, next_state_for)  # 前向
    #         out_weight_for.append(next_state_for[0])
    #         next_state_back = self.backward_lstm(represent, next_state_back)
    #         out_weight_back.append(next_state_back[0])
    #
    #     out_weight_back.reverse()  # 是一个对列表或其他可迭代对象的反转操作
    #
    #     x = self.conv_first(x)  # 卷积    # x: [4,con_fn,480,480]
    #
    #     x1 = x.clone()
    #
    #     for j in range(self.num_block):  # 残差快的块数
    #         # self.lstm_conv_list[j*2] 和 self.lstm_conv_list[j*2+1] 分别是两个 LSTM 单元
    #         out_weight1 = self.lstm_conv_list[j * 2](
    #             torch.cat((out_weight_for[j * 2], out_weight_back[j * 2]), dim=1))  # [128,64, 3,3] 将前向和后向方向的注意力权重拼接在一起
    #         out_weight2 = self.lstm_conv_list[j * 2 + 1](
    #             torch.cat((out_weight_for[j * 2 + 1], out_weight_back[j * 2 + 1]), dim=1))
    #         x1 = self.body[j](x1, out_weight1, out_weight2)     # 用于残差块  out_wight1,2 [128,64,3,3]
    #
    #     x = self.feature_mo(x1)   # 输出得是[4, 2, 480, 480]
    #     return x

# model = CSRNet(in_nc=1, out_nc=1, base_nf=64, cond_nf=32)
# d1 = torch.rand(1, 1, 480, 480)
#
# o = model(d1)
# print()
# onnx_path = "onnx_model_name.onnx"
# torch.onnx.export(model, d1, onnx_path)
# netron.start(onnx_path)


'''
class Same_Conv(nn.Module):
    def __init__(self, chan_fixed, chan_dyn, num_feat, kernel_size):
        super(Same_Conv, self).__init__()
        self.conv1 = HDA_conv(chan_fixed, chan_dyn, num_feat, kernel_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = HDA_conv(chan_fixed, chan_dyn, num_feat, kernel_size)

    def forward(self, x, out_weight1, out_weight2):
        out = self.conv1(x, out_weight1)
        out = self.relu(out)
        out = self.conv2(out, out_weight2)
        # out = self.relu(out)
        return out



class Condition_dyn(nn.Module):
    def __init__(self, num_in_ch, num_feat=32, weight_ratio=0.5, img_range=255.,\
                 kernel_size=3, rgb_mean=(0.4488, 0.4371, 0.4040), num_block=3):
        super(Condition_dyn, self).__init__()

        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.img_range = img_range
        self.num_feat = num_feat
        self.kernel_size = kernel_size  # 卷积核大小
        chan_fixed = int(num_feat * weight_ratio)  # 一半固定的参数
        chan_dyn = num_feat - chan_fixed  # 一半需要改变的参数
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv1 = HDA_conv(chan_fixed, chan_dyn, num_feat, kernel_size)
        self.relu = nn.ReLU()
        self.max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = HDA_conv(chan_fixed, chan_dyn, num_feat, kernel_size)
        self.relu = nn.ReLU()
        self.max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_last = nn.Conv2d(num_feat, num_in_ch, 3, 1, 1)

        self.forward_lstm = ConvLSTMCell((kernel_size, kernel_size), num_feat, num_feat, 1, bias=False)  # 前向
        # (kernel_size, kernel_size)表示输入张量的高度和宽度，num_feat表示输入张量的通道数，也表示hidden_dim隐藏状态的通道数。1表示卷积和大小。
        self.backward_lstm = ConvLSTMCell((kernel_size, kernel_size), num_feat, num_feat, 1, bias=False)  # 后向
        self.lstm_conv_list = nn.ModuleList()
        for _ in range(num_block * 2):  # 前后向之后的卷积  前项和后向各来一遍  就得是num_block*2
            self.lstm_conv_list.append(  # 将Conv2d添加进ModuleList列表中
                nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0)
            )
        self.conv_represent = nn.Sequential(  # 全连接层 DEM之后的那个全连接层
            nn.Linear(4096, num_feat, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(num_feat, chan_dyn * num_feat * kernel_size * kernel_size, bias=False))

        self.num_block = num_block
        self.body = nn.ModuleList()
        for _ in range(num_block):
            self.body.append(
                Same_Conv(   # 一个灰色（HDA）的输出
                    chan_fixed=chan_fixed,
                    chan_dyn=chan_dyn,
                    num_feat=num_feat,
                    kernel_size=kernel_size
                ))
        self.max = nn.AdaptiveAvgPool2d(1)
        self.fc_layer = nn.Linear(64, 2048)

    def forward(self, x, cond):
        # print(x.shape)      # [4, 2, 480, 480]
        # print(cond.shape)   # [4, 1, 4096]

        # cond.size  (1, 4096)
        represent = cond.view(cond.size(0), -1)  # 将输入展平为 [batch_size, 4096]
        # print(represent.shape)  # [4, 4096]
        represent = self.conv_represent(represent)  # 输出
        # print(represent.shape)   # [4, 18432]

        represent = represent.view(-1, self.num_feat, self.kernel_size, self.kernel_size)
        # print(represent.shape)    # [64, 64, 3, 3]  (batch_size, channel, h, w)

        h_init_for = torch.zeros_like(represent)  # 用于LSTM  用于初始化长短时记忆网络（LSTM）状态的张量
        # print('h_init_for', h_init_for.shape)   # [64, 64, 3, 3]
        c_init_for = torch.zeros_like(represent)
        # print('c_init_for', c_init_for.shape)   # [64, 64, 3, 3]
        next_state_for = (h_init_for, c_init_for)
        # print('next_state_for', next_state_for.shape)

        h_init_back = torch.zeros_like(represent)
        # print('h_init_back', h_init_back.shape)   # [64, 64, 3, 3]
        c_init_back = torch.zeros_like(represent)
        # print('c_init_back', c_init_back.shape)   # [64, 64, 3, 3]
        next_state_back = (h_init_back, c_init_back)
        # print('next_state_back', next_state_back.shape)

        out_weight_for = []
        out_weight_back = []

        for i in range(self.num_block * 2):  # 进行LSTM过程
            next_state_for = self.forward_lstm(represent, next_state_for)  # 前向
            out_weight_for.append(next_state_for[0])
            next_state_back = self.backward_lstm(represent, next_state_back)
            out_weight_back.append(next_state_back[0])

        out_weight_back.reverse()  # 是一个对列表或其他可迭代对象的反转操作

        # 对x进行操作
        # self.mean = self.mean.type_as(x)  # 转换mean的数据类型，使其跟x具有相同的数据类型
        # print("self.mean size:", self.mean.size())  # [1, 3, 1, 1]
        # 将输入张量 x 减去均值 self.mean。这样做的目的通常是将数据的均值调整为零，扩展到图像范围
        # x = (x - self.mean) * self.img_range  # x:[4, 2, 480, 480]    mean:[1, 3, 1, 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ]
        # print("x size:", x.size())
        x = self.conv_first(x)  # 卷积
        # print('x1', x.shape)    # [4, 64, 480, 480]             [4, 64, 480, 480]

        x1 = x.clone()
        for j in range(self.num_block):  # 残差快的块数
            # self.lstm_conv_list[j*2] 和 self.lstm_conv_list[j*2+1] 分别是两个 LSTM 单元
            out_weight1 = self.lstm_conv_list[j * 2](
                torch.cat((out_weight_for[j * 2], out_weight_back[j * 2]), dim=1))  # 将前向和后向方向的注意力权重拼接在一起
            out_weight2 = self.lstm_conv_list[j * 2 + 1](
                torch.cat((out_weight_for[j * 2 + 1], out_weight_back[j * 2 + 1]), dim=1))
            x1 = self.body[j](x1, out_weight1, out_weight2)  # 用于残差块   得出

        x = self.max(x)  # 通过最后一个卷积层 conv_last 处理最终的残差连接
        # x = x / self.img_range + self.mean  # 对最终的输出进行反归一化，将数值范围还原，并加上均值
        # print('x2', x.shape)    # [4, 64, 1, 1]   4:batch_size
        return x              # x的输出必须是（某个值，32）


class CSRNet_Dyn(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, base_nf=64, cond_nf=32):
        super(CSRNet_Dyn, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc

        self.cond_net = Condition_dyn(num_in_ch=in_nc)

        self.cond_scale1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.conv1 = nn.Conv2d(in_nc, base_nf, 1, 1, bias=True)

        self.cond_scale2 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift2 = nn.Linear(cond_nf, base_nf, bias=True)
        self.conv2 = nn.Conv2d(base_nf, base_nf, 1, 1, bias=True)

        self.cond_scale3 = nn.Linear(cond_nf, out_nc, bias=True)
        self.cond_shift3 = nn.Linear(cond_nf, out_nc, bias=True)
        self.conv3 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x, cond):
        cond_out = self.cond_net(x, cond)      # 从条件网络得到一个输出
        # print(cond_out.shape)    # [4, 64, 1, 1]
        cond_out = cond_out.view(4, 32)

        # 第一对FC
        scale1 = self.cond_scale1(cond_out)     # Linear层  线性层，用于学习输入特征与输出特征之间的线性映射
        shift1 = self.cond_shift1(cond_out)     # 也是线性层

        # 第二对FC
        scale2 = self.cond_scale2(cond_out)
        shift2 = self.cond_shift2(cond_out)

        # 第三对FC
        scale3 = self.cond_scale3(cond_out)
        shift3 = self.cond_shift3(cond_out)

        out = self.conv1(x)    # 第一个卷积层
        # 这里使用view操作，将scale1和shift1的形状调整为(batch_size, self.base_nf, 1, 1)
        # 也就是相当于  out = out * scale1 + shift1 + out
        out = out * scale1.view(-1, self.base_nf, 1, 1) + shift1.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out)      # 激活层

        out = self.conv2(out)
        out = out * scale2.view(-1, self.base_nf, 1, 1) + shift2.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out)      # 激活层

        out = self.conv3(out)
        out = out * scale3.view(-1, self.out_nc, 1, 1) + shift3.view(-1, self.out_nc, 1, 1) + out
        return out

'''