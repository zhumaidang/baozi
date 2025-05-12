import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩 控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        self.A.weight.data.normal_(mean=0.0, std=0.02)  # 初始化权重
        self.B.weight.data.zero_()  # 初始化偏置

    def forward(self, x):
        return self.B(self.A(x))
    

def apply_lora(model, rank=16):  # 应用LoRA
    for name, module in model.named_modules():  # 遍历模型的所有模块
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:  # 找到线性层
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)  # 创建LoRA网络
            setattr(module, "lora", lora)
            original_forward = module.forward  # 保存原始的前向传播函数

            def forwa