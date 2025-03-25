# 位置编码

## 一、位置编码的演变
1.1 用整型值标记位置
1.2. 用[0, 1]范围标记位置
1.3. 用二进制向量标记位置
1.4. 用周期函数来表示位置
1.5. 用sin和cos交替来表示位置

### 1.1 用整型值标记位置
思路：给第一个token标记1，第二个token标记2，以此类推。
问题：
- 模型可能遇见比训练时所用的序列更长的序列，不利于模型的泛化
- 模型的位置表示是无界的，随着序列长度的增加，位置值会越来越大

### 1.2 用[0, 1]范围标记位置
思路：将每个token的位置值的范围限制再[0, 1]范围内的浮点数。其中，0表示第一个token，1表示最后一个token。
问题：
- 当序列长度不同时，token间的**相对距离**是不一样的。例：三个token[0,0.5,1]，四个token[0,0.33,0.69,1]

需求：
- 它能用来表示一个token在序列中的绝对位置
- 在序列长度不同的情况下，不同序列中token的相对位置/距离也要保持一致
- 可以用来表示模型在训练过程中从来没有看到过的句子长度

### 1.3 用二进制向量标记位置
考虑到位置信息作用在input embedding上，因此比起用单一的值，更好的方案是用一个和input embedding维度一样的向量来表示位置。
| token    | a0  | a1  | a2  | a3  | a4  | a5  | a6 | a7 |
| -------- | --- | --- | --- | --- | --- | --- | --- | --- |
| psoition |  0  |  0  |  0  |  0  |  1  |  1  | 1 | 1 |
| psoition |  0  |  0  |  1  |  1  |  0  |  0  | 1 | 1 |
| psoition |  0  |  1  |  0  |  1  |  0  |  1  | 0 | 1 |

所有的值都是有界的，且取值范围为0或1。且transformer中的d_model足够大，基本可以把每一个位置都编码
问题：
- 该方法位置向量处在一个离散的空间中，不同位置间的变化是不连续的。

需求：
- 将离散空间转换到连续空间，同事不仅能用位置向量表示整形，还可以用位置向量表示浮点型。

### 1.4 用周期函数来表示位置
需要一个有界又连续的函数，正弦函数可以满足这一点。可以考虑把位置向量中的每一个位置都用一个sin函数来表示，则第t个token的位置向量表示为：
$$
PE_t = [\sin(\frac{1}{2^0}t),\sin(\frac{1}{2^1}t)\cdot\cdot\cdot,\sin(\frac{1}{2^{i-1}}t),\cdot\cdot\cdot,\sin(\frac{1}{2^{d_{model}-1}}t)]
$$

结合下图，理解这样设计的含义：
- 图中每一行表示一个$PE_t$，每一列表示$PE_t$中第i个元素，旋钮用于调整精度，越往右边的旋钮，需要调整的精度越大，因此指针移动的步伐越小。每一排的旋钮都在上一排的基础上进行调整（函数中t的作用）。
- 通过频率$\frac{1}{2^{i-1}}$来控制$\sin$函数的波长，频率不断减小，则波长不断增大，此时$\sin$函数对t的变动越不敏感，以此来达到越向右的旋钮，指针游动步伐越小的目的。这也类似于二进制编码，每一位上都是0和1的交互，越往地位走，交互的频率越慢。

![](pic\\周期函数解释.png "周期函数解释") 

![](pic\\二进制编码.jpg "二级制编码") 

问题：
- 由于sin是周期函数，因此从纵向来看，如果函数的频率偏大，引起波长偏短，则不同t下的位置向量可能出现重合的情况。比如在下图中（d_model=3），图中的点表示每个token的位置向量，颜色越深，token的位置越往后，在频率偏大的情况下，位置向量点连成一个闭环，靠前位置（黄色）和靠后位置（棕黑色）靠得非常近：

![](pic\\位置编码重叠问题.jpg "二级制编码") 

为了避免这种情况，尽量将函数的波长拉长。一种简单的解决办法是统一把所有的频率都设置为一个非常小的值。因此在transformer论文中，采用了$\frac{1}{10000^{i/(d_{model}-1)}}$这个频率（这里i其实不是表示第i个位置）。
因此，位置向量的表示为：
$$
PE_t = [\sin(w_0t),\sin(w_1t)\cdot\cdot\cdot,\sin(w_{i-1}t),\cdot\cdot\cdot,\sin(w_{d_{model}-1}t)]
$$
其中，$w_i=\frac{1}{10000^{i/(d_{model}-1)}}$

### 1.5 用sin和cos交替来表示位置
目前位置，我们的位置向量实现了如下功能：
1. 每个token的向量唯一（每个sin函数的频率足够小）
2. 位置向量的值是有界的，且位于连续空间中。模型在处理位置向量时更容易泛化，即更好处理长度和训练数据分布不一致的序列（sin函数本身的性质）

需求：
- 不同位置向量是可以通过线性转换得到的。这样不仅能表示一个token的绝对位置，还可以表示一个token的相对位置，即：
$$
PE_{t+\bigtriangleup t} = T_{\bigtriangleup t} * PE_{t}
$$

这里，T表示一个线性变换矩阵。观察目标式子，联想到在向量空间中一种常见的线性变换--旋转。在这里将t想象为一个角度，那么$\bigtriangleup t$就是其旋转的角度，则上面的式子可以进一步写成：
$$
\begin{pmatrix}
\sin(t+\bigtriangleup t)\\
\cos(t+\bigtriangleup t)\\
\end{pmatrix}
=
\begin{pmatrix}
\cos\bigtriangleup t & \sin \bigtriangleup t\\
-\sin \bigtriangleup t & \cos\bigtriangleup t\\
\end{pmatrix}
\begin{pmatrix}
\sin t\\
\cos t\\
\end{pmatrix}
$$

有了这个构想，可以把原来元素全都是sin函数的$PE_t$做一个替换，让位置两两一组，分别用sin和cos函数表示，则：
$$
PE_t = [\sin(w_0t),\cos(w_0t),\sin(w_1t),\cos(w_1t),\cdot\cdot\cdot,\sin(w_{\frac{d_{model}}{2}-1}t),\cos(w_{\frac{d_{model}}{2}-1}t)]
$$

在这样的表示下，很容易用一个线性变换，把$PE_t$转变为$PE_{t+\bigtriangleup t}$:
$$
PE_{t+\bigtriangleup t} = T_{\bigtriangleup t} * PE_t = 
\begin{pmatrix}
\begin{bmatrix}
\cos(w_0\bigtriangleup t) & \sin (w_0\bigtriangleup t)\\
-\sin(w_0\bigtriangleup t) & \cos(w_0\bigtriangleup t)\\
\end{bmatrix} & ... &  0 \\
... & ... & ... \\
0 &... & 
\begin{bmatrix}
\cos(w_{\frac{d_{model}}{2}-1}\bigtriangleup t) & \sin (w_{\frac{d_{model}}{2}-1}\bigtriangleup t)\\
-\sin(w_{\frac{d_{model}}{2}-1}\bigtriangleup t) & \cos(w_{\frac{d_{model}}{2}-1}\bigtriangleup t)\\
\end{bmatrix}

\end{pmatrix}
\begin{pmatrix}
sin(w_0t)\\
cos(w_0t)\\
...\\
sin(w_{\frac{d_{model}}{2}-1}t)\\
cos(w_{\frac{d_{model}}{2}-1}t)\\
\end{pmatrix}=
\begin{pmatrix}
sin(w_0(t+\bigtriangleup t))\\
cos(w_0(t+\bigtriangleup t))\\
...\\
sin(w_{\frac{d_{model}}{2}-1}(t+\bigtriangleup t))\\
cos(w_{\frac{d_{model}}{2}-1}(t+\bigtriangleup t))\\
\end{pmatrix}
$$

## 二、Transformer中位置编码方法：Sinusoidal functions
2.1. 位置编码定义
2.2. 位置编码可视化
2.3. 位置编码重要性质

### 2.1 位置编码定义
有了上面的演变过程，观察transformer中的位置编码方法。
定义：
- t是这个token在序列中的实际位置（例如第一个token为1，第二个token为2，以此类推）
- $PE_t \in \mathbb{R}^d$是这个token的位置向量，PE_{t}^{(i)}表示这个位置向量里的第i个元素
- d_{model}是这个token的维度

则PR_t^{(i)}可以表示为：
$$
PE_t^{(i)} = 
\begin{cases}
\sin(w_kt), &  if & i=2k \\
\cos(w_kt), &  if &  i=2k+1 \\
\end{cases}
$$

其中：
$$
w_k = \frac{1}{10000^{2k/d_{model}}} \\
$$

$$
i = 0, 1, 2, ..., \frac{d_{model}}{2} - 1 
$$

### 2.2 位置编码可视化
下图是一串序列长度为50，位置编码维度为128的位置编码可视化结果

![](pic\\位置编码可视化.jpg "二级制编码") 

可以发现，由于sin/cos函数的性质，位置向量的每一个值都位于[-1,1]之间。同事，纵向来看，图的右半边几乎都是蓝色的，这是因为越往后的位置，频率越小，波长越长，所以不同的t对最终的结果影响不大。而越往左边走，颜色交替的频率越频繁。

### 2.3 位置编码重要性质
性质一：两个位置编码的点积仅取决于偏移量$\bigtriangleup t$，也即两个位置编码的点积可以反应出两个位置编码间的距离。

性质二：位置编码的点积是无向的，即$PE_t^T*PE_{t+\bigtriangleup t}=PE_t^T * PE_{t-\bigtriangleup t}$

# 参考
<https://zhuanlan.zhihu.com/p/454482273



