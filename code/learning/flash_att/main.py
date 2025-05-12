'''
 # @ Author: zhu jiangqiang
 # @ Create Time: 2025-04-23 11:04:49
 # @ Modified by: zhu jiangqiang
 # @ Modified time: 2025-04-23 11:06:35
 # @ Description:
 '''

#%% 
import torch

# %%
NEG_INF = -1e10  # 极小数
EPSILON = 1e-10  # 极大数

Q_LEN = 6  # query长度
K_LEN  =6  # key长度
Q_BLOCK_SIZE = 3  # query块大小
KV_BLOCK_SIZE = 3  # key-value块大小  Q块的大小于KV块的大小可以不相等
P_DROP = 0.2  # drop概率

# %%
Tr = Q_LEN // Q_BLOCK_SIZE  # query块的数量
Tc = K_LEN // KV_BLOCK_SIZE  # key-value块的数量
print("Tr:", Tr, "Tc:", Tc)

# %%
Q = torch.randn(1, 1, Q_LEN, 4, requires_grad=True).to(device='cpu')  # batch_size,num_heads,q_len,dim
K = torch.randn(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')
V = torch.randn(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')
print("Q:", Q, "\n")
print("Q:", K, "\n")
print("Q:", V, "\n")

# %%
O = torch.zeros_like(Q, requires_grad=True)
l = torch.zeros(Q.shape[:-1])[..., None]   # [..., None]在最后添加一个维度
m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF
print("O:", O, "\n")
print("l:", l, "\n")
print("m:", m, "\n")

# %%
Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)

O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

# %%
for j in range(Tc):
    Kj = K_BLOCKS[j]
    Vj = V_BLOCKS[j]

    for i in range(Tr):
        Qi = Q_BLOCKS[i]
        Oi = O_BLOCKS[i]
        li = l_BLOCKS[i]
        mi = m_BLOCKS[i]

        S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi, Kj)
        print("===>>>S_ij  shape:", S_ij.shape)

        mask = S_ij.ge(0.5)  # 返回一个与S_ij形状相同的布尔张量，其中每个元素表示对应位置的值是否大于等于0.5
        S_ij = torch.masked_fill(S_ij, mask, value=0)

        m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
        P_ij = torch.exp(S_ij - m_block_ij)
        l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON
        P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

        # 更新m, l
        mi_new = torch.maximum(m_block_ij, mi)
        li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij

        m = torch.nn.Dropout(p=P_DROP)
        P_ij_Vj = m(P_ij_Vj)

        O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Qi + (torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj

        print(f'-----------Attention:Q{i}*K{j}------------')
        print(O_BLOCKS[i].shape)
        print(O_BLOCKS[0])
        print(O_BLOCKS[1])
        print('\n')

        l_BLOCKS[i] = li_new
        m_BLOCKS[i] = mi_new

O = torch.cat(O_BLOCKS, dim=2)
l = torch.cat(l_BLOCKS, dim=2)
m = torch.cat(m_BLOCKS, dim=2)

# %%
