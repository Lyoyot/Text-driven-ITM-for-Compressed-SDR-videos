import torch

from torch import nn, einsum

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n, C, b = x.size()
        query = self.query_conv(x).view(b, -1, n).permute(0, 2, 1)  # [batch_size, HW, C//8]
        key = self.key_conv(x).view(b, -1, n)  # [batch_size, C//8, HW]
        value = self.value_conv(x).view(b, -1, n)  # [batch_size, C, HW]
        attention_scores = torch.bmm(query, key)  # [batch_size, HW, HW]
        attention_scores = self.softmax(attention_scores)  # [batch_size, HW, HW]
        attention_output = torch.bmm(value, attention_scores.permute(0, 2, 1))  # [batch_size, C, HW]
        attention_output = attention_output.view(b, C, n)  # [batch_size, C, H, W]
        out = self.gamma * attention_output + x
        return out

class TextSelfAttention(nn.Module):
    def __init__(self, text_dim):
        super(TextSelfAttention, self).__init__()
        self.query_fc = nn.Linear(text_dim, text_dim)
        self.key_fc = nn.Linear(text_dim, text_dim)
        self.value_fc = nn.Linear(text_dim, text_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query_fc(x)
        key = self.key_fc(x)
        value = self.value_fc(x)
        attention_scores = torch.bmm(query.unsqueeze(1), key.unsqueeze(1).permute(0, 2, 1))
        attention_scores = self.softmax(attention_scores)
        attention_output = torch.bmm(attention_scores, value.unsqueeze(1)).squeeze(1)
        out = self.gamma * attention_output + x
        return out

class CrossAttention(nn.Module):
    def __init__(self, img_channels, text_dim, out_channels):
        super(CrossAttention, self).__init__()

        self.self_attention = SelfAttention(img_channels)
        self.text_self_attention = TextSelfAttention(text_dim)
        self.query_conv = nn.Conv2d(img_channels, out_channels, kernel_size=1)
        self.key_fc = nn.Linear(text_dim, out_channels)
        self.value_fc = nn.Linear(text_dim, out_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, img_feat, text_feat):  # [b,c,480,480]  text:[b, 512]
        batch_size, C, H, W = img_feat.size()

        # Apply self-attention to image features
        # img_feat = self.self_attention(img_feat)

        # Text self-attention
        # text_feat = self.text_self_attention(text_feat)

        # Query from image features   通过卷积层将image_feat转换为查询向量，并重塑为[b,H*W,out_c]
        query = self.query_conv(img_feat).view(batch_size, -1, H * W).permute(0, 2, 1)  # [B, HW, out_channels]

        # Key and Value from text features  计算键向量key，并给其增加一个维度
        key = self.key_fc(text_feat).unsqueeze(1)  # [B, 1, out_channels]
        value = self.value_fc(text_feat).unsqueeze(1)  # [B, 1, out_channels]

        # query = self.key_fc(text_feat).unsqueeze(1)  # [B, 1, out_channels]   (4, 1, 64)
        #
        # key = self.query_conv(img_feat).view(batch_size, -1, H * W).permute(0, 2, 1)  # [B, HW, out_channels]  (4, 230400, 64)
        # value = self.query_conv(img_feat).view(batch_size, -1, H * W).permute(0, 2, 1)  # [B, HW, out_channels]   (4, 230400, 64)

        # Compute attention scores
        attention_scores = torch.bmm(query, key.transpose(1, 2))  # [B, HW, 1]  really:(b, 1, HW)  通过批量矩阵乘法‘torch.bmm’计算q和k的内积
        attention_scores = attention_scores - attention_scores.max(dim=-1, keepdim=True)[0]
        attention_scores = self.softmax(attention_scores)  # [B, HW, 1]    将得分归一化

        attention_scores_sum = attention_scores.sum(dim=-1, keepdim=True) + 1e-10
        attention_scores = attention_scores / attention_scores_sum

        # Apply attention scores to value
        attention_output = torch.bmm(attention_scores, value).view(batch_size, H, W, -1)  # [B, H, W, out_channels]
        attention_output = attention_output.permute(0, 3, 1, 2)  # [B, out_channels, H, W]

        # Combine with original image features
        out = self.gamma * attention_output + img_feat

        return out, attention_output