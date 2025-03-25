
import torch
import torch.nn as nn

embedding = nn.Embedding(10, 3)  # 词表大小  词向量维度
input = torch.LongTensor([[1,2,4,5,1],[4,3,2,9,1]])
emb = embedding(input)
print(emb)
print(emb.shape)


