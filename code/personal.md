

# 数据集
| 数据集名称 | 样本数量 | 大小 | 数据来源 | 领域 |
|:---------:|:---------:|:---------:|:---------:|:---------:|
| pretrain_hq.jsonl | 1413120 | 1.6GB | 匠数科技 | 未知 |

# 预训练阶段
- [x] 预训练后的模型会有复读现象
- [x] 分布式训练DDP
- [x] 位置旋转编码（RoPE：既包含相对位置信息也包含绝对位置信息）
- [x] BPE分词器、BBPE
- [x] 混合专家网络
- [x] flash attention
