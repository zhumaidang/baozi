import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    """
    LoRA 网络结构，用于在Transformer模型中添加低秩矩阵。

    lora目的：用一个低秩线性变换来近似学习原始线性层的微调残差
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩 控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        self.A.weight.data.normal_(mean=0.0, std=0.02)  # 正态分布初始化权重
        self.B.weight.data.zero_()  # 初始化偏置  初始化为0，使得一开始不会影响原始网络。如果对B矩阵进行随机初始化，初始模型行为呗扰乱，可能导致性能骤降或训练不稳定。

    def forward(self, x):
        return self.B(self.A(x))
    

def apply_lora(model, rank=8):  # 应用LoRA
    for name, module in model.named_modules():  # 遍历模型的所有模块
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:  # 找到线性层
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)  # 创建LoRA网络
            setattr(module, "lora", lora)  # 动态得给module对象添加一个"lora"属性，并赋值为lora对象
            original_forward = module.forward  # 保存原始的前向传播函数

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)  # 前向传播函数，将原始的前向传播函数和LoRA网络结合

            module.forward = forward_with_lora  # 将新的前向传播函数赋值给模块的前向传播函数

    
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device) 
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''):v for k,v in state_dict.itmes() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {f'{name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
