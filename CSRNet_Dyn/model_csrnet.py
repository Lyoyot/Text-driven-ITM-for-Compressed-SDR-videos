import CSRNet_Dyn.CSRNet_arch as CSRNet_arch

# import CSRNet_Dyn.CSRNet_arch_Fc as CSRNet
import torch.nn as nn
import torch

from data.ICTCP_convert import SDR_to_ICTCP, ICTCP_to_HDR

'''
class WholeNet(nn.Module):
    def __init__(self, device, num=3, channel=32):
        super(WholeNet, self).__init__()
        self.CSRNet_I = CSRNet_arch.CSRNet(in_nc=1, out_nc=1, base_nf=64, cond_nf=32)
        self.CSRNet_TP = CSRNet_arch.CSRNet_Dyn(in_nc=2, out_nc=2, base_nf=64, cond_nf=32)

    def forward(self, sdrRGB, cond):   # text_input: [4, 77]

        sdrITP = SDR_to_ICTCP(sdrRGB,dim=1)
        sdrI, sdrT, sdrP = torch.split(sdrITP, 1, dim=1)
        sdrTP = torch.cat([sdrT, sdrP], dim=1)
        I = self.CSRNet_I(sdrI)   # [4, 1, 480, 480]  全局？

        TP = self.CSRNet_TP(sdrTP, cond)

        ITP = torch.cat([I, TP], dim=1)    # 直接输出？
        hdrRGB = ICTCP_to_HDR(ITP,dim=1)

        return [ITP, hdrRGB]
'''
'''
class WholeNet(nn.Module):
    def __init__(self, device, num=3, channel=32):
        super(WholeNet, self).__init__()
        self.CSRNet_I = CSRNet_arch.CSRNet(in_nc=1, out_nc=1, base_nf=64, cond_nf=32)
        self.CSRNet_TP = CSRNet_arch.CSRNet(in_nc=2, out_nc=2, base_nf=64, cond_nf=32)

    def forward(self, sdrRGB):   # text_input: [4, 77]

        sdrITP = SDR_to_ICTCP(sdrRGB,dim=1)
        sdrI, sdrT, sdrP = torch.split(sdrITP, 1, dim=1)
        sdrTP = torch.cat([sdrT, sdrP], dim=1)
        I = self.CSRNet_I(sdrI)   # [4, 1, 480, 480]  全局？

        TP = self.CSRNet_TP(sdrTP)

        ITP = torch.cat([I, TP], dim=1)    # 直接输出？
        hdrRGB = ICTCP_to_HDR(ITP,dim=1)

        return [ITP, hdrRGB]
'''
class DimensionalityTransformer(nn.Module):
    def __init__(self, input_shapes, target_dim):
        super(DimensionalityTransformer, self).__init__()
        self.target_dim = target_dim

        # 为每个输入张量定义一个全连接层
        self.fc_layers = nn.ModuleList([
            nn.Linear(in_dim, target_dim) if in_dim != target_dim else nn.Identity()
            for in_dim in input_shapes
        ])

    def forward(self, inputs):
        # 输入 inputs 是一个张量列表
        converted_inputs = [fc(tensor) for fc, tensor in zip(self.fc_layers, inputs)]
        return torch.cat(converted_inputs, dim=1)

class WholeNet(nn.Module):
    def __init__(self, model, device, num=3, channel=32):
        super(WholeNet, self).__init__()
        self.model = model
        # 冻结语言模型参数
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # self.CSRNet_I = CSRNet_arch.CSRNet_Dyn(in_nc=1, out_nc=1, base_nf=64, cond_nf=32)
        # self.CSRNet_TP = CSRNet_arch.CSRNet_Dyn(in_nc=2, out_nc=2, base_nf=64, cond_nf=32)
        self.CSRNet_I = CSRNet_arch.CSRNet_Dyn(in_nc=1, out_nc=1, base_nf=64, cond_nf=32)
        self.CSRNet_TP = CSRNet_arch.CSRNet_Dyn(in_nc=2, out_nc=2, base_nf=64, cond_nf=32)

    def forward(self, sdrRGB, text):
        # print(sdrRGB.dtype)
        # print(text.dtype)

        # with torch.no_grad():
        #     text_degra = self.model(degra, return_tensors='pt', add_special_tokens=False).to("cuda")   # 输出：[4, 512]
        #     text_content = self.model(content, return_tensors='pt', add_special_tokens=False).to("cuda")  # 输出：[4, 512]
        #     text_quality = self.model(quality, return_tensors='pt', add_special_tokens=False).to("cuda")  # 输出：[4, 512]
            # text_feature2 = self.clip_model.encode_text(text_input2)

        # concatenated_text = torch.cat((degra, content, bright, color, detail), dim=1)   # 输出: torch.Size([4, 12288])

        text = text.to(torch.float32)
        sdrITP = SDR_to_ICTCP(sdrRGB,dim=1)
        sdrI, sdrT, sdrP = torch.split(sdrITP, 1, dim=1)
        sdrTP = torch.cat([sdrT, sdrP], dim=1)
        # text_feature = torch.concat(text_feature1, text_feature2)
        # I = self.CSRNet_I(sdrI, text_feature.float())   # [4, 1, 480, 480]  全局？
        I = self.CSRNet_I(sdrI, text)

        TP = self.CSRNet_TP(sdrTP, text)

        ITP = torch.cat([I, TP], dim=1)    # 直接输出？
        hdrRGB = ICTCP_to_HDR(ITP,dim=1)

        return [ITP, hdrRGB]