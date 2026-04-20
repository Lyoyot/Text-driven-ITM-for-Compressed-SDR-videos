import torch.nn as nn
import torch
from einops import rearrange, repeat
import torch.nn.functional as F


# 多头注意力  来源 Spatial-Frequency Mutual Learning for Face Super-Resolution
class Attention(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):   # 输入维度
        super(Attention, self).__init__()
        self.num_heads = num_heads       # 头数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))    # 一个可训练参数， 用于调整注意力分数的尺度，帮助稳定训练过程

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)     # 用于生成键值
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)   # 创建一个深度可分离卷积层，用于进一步处理键和值
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)        # 创建一个1x1卷积，用于生成查询query
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)   # 也是用于生成可分离卷积
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape    # 获取b,c,h,w

        kv = self.kv_dwconv(self.kv(y))     # 对输入y进行键值生成和深度卷积处理
        k, v = kv.chunk(2, dim=1)           # 将kv按照通道数拆分为k和v
        q = self.q_dwconv(self.q(x))        # 对输入x进行查询生成和深度卷积处理

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)   # 将查询q，k，v重塑为多头注意力所需的形状
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)     # 对q，k进行归一化处理
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature   # 计算查询和键的点积，并乘以温度参数 self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)    # 将注意力分数与值 v 相乘，得到加权后的输出

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)   # 重塑回原始形状

        out = self.project_out(out)     # 通过 self.project_out 层对输出进行处理
        return out

class FuseBlock7(nn.Module):
    def __init__(self, channels):
        super(FuseBlock7, self).__init__()
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fre = nn.Linear(512, channels * 480 * 480)
        self.fre_att = Attention(dim=channels)
        self.spa_att = Attention(dim=channels)
        self.fuse = nn.Sequential(nn.Conv2d(2*channels, channels, 3, 1, 1), nn.Conv2d(channels, 2*channels, 3, 1, 1), nn.Sigmoid())

    def forward(self, spa, fre):
        # ori = spa
        fre = self.fre(fre).view(fre.size(0), -1, 480, 480)  # 调整 fre 的形状以便进行卷积操作
        spa = self.spa(spa)     # 先经过卷积
        fre = self.fre_att(fre, spa)+fre
        spa = self.fre_att(spa, fre)+spa             # 再经过attention
        fuse = self.fuse(torch.cat((fre, spa), 1))   # 再进行融合
        fre_a, spa_a = fuse.chunk(2, dim=1)
        spa = spa_a * spa
        fre = fre * fre_a
        res = fre + spa

        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res