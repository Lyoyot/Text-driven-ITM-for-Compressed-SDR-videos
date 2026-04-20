
# 这个attention来自stable_diffusion
# 作用：从一个查询（query）向量中提取关于上下文（context）向量的注意力信息，从而在输入数据之间进行信息融合
import torch
from torch import nn, einsum
from inspect import isfunction
from einops import rearrange, repeat
import torch.nn.functional as F
# CrossAttn precision handling
import os
from typing import Optional, Any
# 环境变量，用于控制注意力计算时的精度
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


def exists(val):    # 用于检查是否为验证阶段
    return val is not None

def default(val, d):    # 用于返回默认值，如val存在，则返回val，否则返回d或调用d函数
    if exists(val):
        return val
    return d() if isfunction(d) else d

def safe_softmax(x, dim=-1, epsilon=1e-10):
    """
    计算安全的 Softmax，防止除数为零。
    :param x: 输入张量
    :param dim: 计算 Softmax 的维度
    :param epsilon: 防止除数为零的小常数
    :return: Softmax 结果
    """
    # 计算输入张量的最大值，用于数值稳定性
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - max_val)
    sum_x_exp = torch.sum(x_exp, dim=dim, keepdim=True) + epsilon  # 增加 epsilon 防止除数为零
    return x_exp / sum_x_exp

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class CrossAttention_SD(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):  # 定义了注意力头数’heads‘，每个头维度为dim_head，以及其他的相关的线性变换层
        super().__init__()
        inner_dim = dim_head * heads   # 注意力头总维度
        context_dim = default(context_dim, query_dim)   # 默认等于query_dim， 如果未指定，则上下文和查询维度相同

        self.scale = dim_head ** -0.5   # 缩放因子，用于稳定梯度，缩放点积注意力
        self.heads = heads      # 头数

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)    # 分别是将query key value变换到多头注意力的线性层
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )                                # 是输出的线性变换和dropout层

    def forward(self, x, context=None, mask=None):
        h = self.heads     # 头部数量

        q = self.to_q(x)      # 计算查询q
        context = default(context, x)     # 设置上下文，如果未提供则使用输入x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))   # 重排qkv 以适应多头注意力

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":   # 如果这个值为32，则将q，k强制转换为fp32，防止溢出
            with torch.autocast(enabled=False, device_type='cuda'):   #
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale    # 计算注意力分数sim

        del q, k

        if exists(mask):   # 如果存在mask，则应用掩码来屏蔽无效的位置
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = safe_softmax(sim, dim=-1)                #  对注意力分数进行 softmax 归一化

        out = torch.einsum('b i j, b j d -> b i d', sim, v)   # 计算加权值 v
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)      # 重排输出张量
        return self.to_out(out)

# 实现了一种高效的交叉注意力机制
class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class CrossAttention_Text(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            with torch.autocast(enabled=False, device_type='cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = safe_softmax(sim, dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class SelfAttention_SD(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(SelfAttention_SD, self).__init__()
        inner_dim = dim_head * heads  # 内部维度是注意力头数与每个头的维度的乘积
        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子，用于稳定梯度

        # 将输入特征映射到查询、键、值的线性变换
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        # 输出层，包括线性变换和dropout
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        h = self.heads

        # 计算查询、键、值
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # 将q、k、v重排以适应多头注意力
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # 计算注意力分数
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        # 如果存在mask，则应用掩码
        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim = sim.masked_fill(~mask, max_neg_value)

        # 对注意力分数进行softmax归一化
        sim = safe_softmax(sim, dim=-1)

        # 计算加权值
        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)


class SelfAttention_Text(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(SelfAttention_Text, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        assert (
                self.head_dim * num_heads == in_channels
        ), "in_channels must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)

        self.out = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        # x has shape (b, c)
        b, c = x.shape
        # Add sequence length dimension for self-attention
        x = x.unsqueeze(1)  # Shape: (b, 1, c)
        x = x.repeat(1, c, 1)  # Shape: (b, c, c)

        # Split into multiple heads
        q = self.query(x).view(b, c, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, c, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, c, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = safe_softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(b, c, c)
        out = self.out(out)

        # Only return the output of the first token (or sequence length 1)
        out = out[:, 0]  # Shape: (b, c)

        return out

class ImageTextFusion(nn.Module):
    def __init__(self, image_dim, text_dim=512, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.flatten_dim = image_dim[1] * image_dim[2] * image_dim[3]

        self.cross_attention = CrossAttention_SD(self.flatten_dim, text_dim, heads, dim_head, dropout)

    def forward(self, image_features, text_features):
        # Flatten image features
        b, c, h, w = image_features.shape
        image_features = image_features.view(b, c * h * w).unsqueeze(1)  # shape: [b, 1, c*h*w]

        # Expand text features
        text_features = text_features.unsqueeze(1).repeat(1, image_features.size(2), 1)  # shape: [b, c*h*w, text_dim]

        # Perform cross attention
        fused_features = self.cross_attention(image_features, text_features)

        return fused_features
