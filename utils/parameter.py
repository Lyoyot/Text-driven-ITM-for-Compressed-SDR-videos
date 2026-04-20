import torch
from LCATNet.model_fusion_new import WholeNet_new
# 假设你已经定义好了模型结构
model = WholeNet_new()

# 加载权重文件
checkpoint_path = r"D:\pycharm-workspace\ITPNet_neew\LCATNet\model_alldata_model28_new_with_one_cross\model_150.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 分别加载子模块参数
model.CSRNet_I.load_state_dict(checkpoint['CSRNet_I'])
model.CSRNet_TP.load_state_dict(checkpoint['CSRNet_TP'])
model.fusionNet_new.load_state_dict(checkpoint['fusionNet_new'])

# 设置为评估模式（可选）
model.eval()

# 参数统计函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 输出参数数量
print(f"总参数数目: {count_parameters(model)}")
print(f"可训练参数数目: {count_trainable_parameters(model)}")
