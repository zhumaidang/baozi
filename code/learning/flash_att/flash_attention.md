# Flash Attention

## 一、概念

Flash Attention旨在**加速注意力计算**并**减少内存占用**。

1. 利用底层硬件的内存层次知识（GPU），来提高计算速度和内存访问开销
2. 核心原理：通过将输入分块并在每个块上执行注意力操作，从而减少对高带宽内存（HBM）的读写操作
3. Flash Attention使用平铺和重计算等经典技术，将输入块从HBM加载到SRAM（快速缓存），在SRAM上执行注意力操作，并将结果更新回HBM。
4. FlashAttention减少了内存读写量，从而实现了2-4倍的时钟时间加速。
5. Flash Attention-2版本进一步优化了Flash Attention算法，使用了更好的并行化和工作分区方法，使得计算速度提高了2倍。
6. Flash Attention-2还支持更高的头维数和多查询注意力

### 1.1 为什么需要Attention加速

Transformer中self-attention的**时间和内存复杂度是序列长度的二次方**，所以序列过长时，算法速度会变慢，需要消耗很高的内存。

> 时间复杂度O(N*N)：attention需要对矩阵Q和K做矩阵运算，时间复杂度是序列长度n的平方级，即attention的时间复杂度为O(n\*n)。当序列较长时，attention的计算非常耗时。
>
> 空间复杂度O(N*N)：attention的计算过程需要存储S和P这两个尺寸均为(n,n)的矩阵，因此，attention运算的空间复杂度也为O(N\*N)

### 1.2 HBM和SRAM

HBM（High Bandwidth Memory)：1.5TB/s

SRAM(Static Random-Access Memory)：19TB/s

两种不同类型的计算机内存：

- HBM是一种高带宽内存接口，用于3D堆叠的SDRAM，具有较高的带宽和较低的功耗
- SRAM是一种静态随机访问存储器，用于高速缓存等内部存储器，具有更快的访问速度和更低的延迟，但成本更高且占用更多芯片空间。

MAC（Memory Access Cost，存储访问开销）：
指在计算机系统中，访问内存或存储器所需的时间和资源开销，它是衡量计算机程序或算法性能的重要指标之一。MAC的值取决于多个因素，包括内存层次结构，缓存命中率，内存带宽，存储器延迟等。较低的MAC值表示访问内存的开销较小，而较高的MAC值表示访问内存的开销较大。

## 二、原理

### 2.1 传统的Attention

对于输入序列
$$
Q,K,V\in \mathbb{R}^{N\times d}
$$
其中，N是序列长度，d是toke维度，N远大于d，self-attention计算输出
$$
O\in \mathbb{R}^{N\times d}
$$
计算过程如下：
$$
S = QK^T\in \mathbb{R}^{N\times N} \\
P = softmax(S)\in \mathbb{R}^{N\times N} \\
O = PV\in \mathbb{R}^{N\times d}
$$
标准注意力实现将矩阵S和P具体化为HBM，这需要$O(N^2)$内存，传统attention的流程如图：

![img](https://pic1.zhimg.com/v2-be1b807aff7e52ace47eeeb800f3a42e_1440w.jpg)

### 2.2 FlashAttention算法

**核心思想**：传统减少HBM的访问，将QKV切分为小块后放入SRAM中

**核心方法**：tiling，recomputation

#### 2.2.1 tiling（平铺）：分块计算

> 不直接对整个输入序列计算注意力，而是将其分为多个较小的块，逐个对这些块进行计算，增量式的进行softmax的规约，规约过程中只需要更新某些中间变量，不需要计算整个注意力权重矩阵

attention计算中设计softmax，所以不能简单的分块后直接计算

之前softmax计算方法：
$$
softmax(x_j)=\frac{x^{x_j}}{\sum^{k}_{i=1}e^{x_i}}
$$
softmax的操作是row-wise的，即每行都算一次softmax，所以需要用到平铺算法来分块计算softmax。

safe softmax：原始softmax数值不稳定，为了数值稳定性，Flash Attention采用safe softmax，向量$\in R$的safe softmax计算如下：

> softmax数值不稳定的原因：输入值过大或过小导致溢出。
>
> softamx中指数函数exp(x)对数值非常敏感，输入值过大可能会导致：
>
> 1. exp(x)结果非常大，导致数值溢出（变成inf或NaN）；
>
> 2. 反过来，极小的输入可能变成0，造成精度问题。
>
> 解决方案：减去最大值

$$
m(x):=max_ix_i \\
f(x):=[e^{x1-m(x)} ...e^{x_B-m(x)}] \\
\ell(x):=\sum_if(x)_i \\
softmax(x):=\frac{f(x)}{\ell(x)}
$$
f(x)h和l(x)都可以通过分块计算得出，所以flashattention在计算时通过分块将Q，K，V分块后，按块加载到内存中。

#### 2.2.2 recomputation（重新计算）

Flash Attention算法的目标：在计算中减少显存占用，从O(N*N)大小降到线性，这样就可以把数据加载到SRAM中，提高IO速度。

解决方案：传统attention在计算中需要用到Q，K，V去计算S，P两个矩阵，FlashAttention引入softmax中的统计量（m,l），结合output O和在SRAM中的Q，K，V块进行计算。

![img](https://pic2.zhimg.com/v2-a3e7f759b3ae9123fd0d7034f2ee6d01_1440w.jpg)







[手撕Flash Attention！原理解析及代码实现🦥 - 知乎](https://zhuanlan.zhihu.com/p/696850636)

[Flash Attention原理详解(含代码讲解) - 知乎](https://zhuanlan.zhihu.com/p/676655352)
