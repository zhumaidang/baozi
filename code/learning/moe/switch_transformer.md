# Switch Transformer

switch transformer接收两个输入（两个不同的token）并拥有四个专家。与最初使用至少两个专家的想法相反，switch transformer采用了简化的单专家策略。效果：

1. 减少门控网络设计负担
2. 每个专家的批量大小至少可以减半
3. 降低通信成本
4. 保持模型质量



专家容量
$$
Expert Capacity=(\frac{token\quad per\quad batch}{number\quad of\quad experts})\times capacity\quad factor
$$


![img](https://picx.zhimg.com/v2-8893a3a28e45d0c2210030b9f42d8191_1440w.jpg)



## 总结

混合专家模型：

1. 与稠密模型相比，**预训练速度更快**
2. 与具有相同参数数量的模型相比，具有更快的**推理速度**
3. 需要**大量显存**，因为所有专家系统都需要加载到内存中
4. 在微调方面存在诸多挑战，但近期的研究表明，对混合专家模型进行指令调优具有很大的潜力









