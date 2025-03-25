

# Tips
旋转位置编码（Rotray Position Embedding, RoPE）提出了一种能够将相对位置信息依赖集成到self-attention中并提升transformer架构性能的位置编码方式。
和相对位置编码相比，RoPE具有更好的外推性，目前是大模型相对位置编码中应用最广的方式之一。

> 大模型的外推性
外推性是指大模型再训练时和预测时的输入长度不一致，导致模型的泛化能力下降的问题。例如，如果一个模型在训练时只使用了512个token的文本，那么在预测时如果输入超过512个token，模型可能无法正确处理。

# 旋转位置编码RoPE
## 1.1 基本概念
定义一个长度为N的输入序列为：
$$
\mathbb{S} = \{{w_i}\}_{i=1}^N
$$

其中，$w_i$表示输入序列中第i个token，而输入序列$\mathbb{S}_N$对应的embedding，表示为：
$$
\mathbb{E}_N = \{{x_i}\}_{i=1}^N
$$

其中$x_i$表示第i个token $w_i$对应的d维词嵌入向量。
接着在做self-attention之前，会用词嵌入向量计算q,k,v向量同事加入位置信息，函数公式表示如下：
$$
q_m = f_q(x_m, m)  \\
k_n = f_k(x_n, n)  \\
v_n = f_v(x_n, n)
$$

其中$q_m$表示第m个token对应的词向量$x_m$集成位置信息m之后的query向量。而$k_n$和$v_n$则表示第n个token对应的词向量$x_n$集成位置信息n之后的key和value向量。
而基于transformer的位置编码方法都是着重于构造一个合适的$f(q,k,v)$函数形式。

计算第m个词嵌入向量$x_m$对应的self-attention输出结果，就是将$q_m$和其他$k_n$都计算一个attention score，然后再将attention score乘以对应的$v_n$再求和得到输出向量$o_m$:
$$
a_{m,n} = \frac{exp(\frac{q_m^Tk_n}{\sqrt{d}})}{\sum_{j=1}^{N}exp(\frac{q_m^Tk_j}{\sqrt{d}})}  \\
o_m = \sum_{n=1}^{N}a_{m,n}v_n
$$

## 1.2 绝对位置编码
常规做法时再计算query，key和value向量之前，会计算一个位置编码向量$p_i$加到词嵌入$x_i$上，位置编码向量$p_i$同样也是d维向量，然后再乘以对应的变换矩阵W:
$$
f_{t:t\in \{q,k,v\}}(x_i, i) := W_{t:t\in \{q,k,v\}(x_i + p_i)}
$$

经典的位置编码向量$p_i$的计算方式使用Sinusoidal函数：
$$
p_{i, 2t} = \sin(\frac{k}{10000^{\frac{2t}{d}}}) \\
p_{i, 2t+1} = \cos(\frac{k}{10000^{\frac{2t}{d}}})
$$

其中$p_{i, 2t}$表示偶数索引位置的计算公式，$p_{i, 2t+1}$就对应奇数索引位置的计算公式。


## 1.3 二维旋转位置编码
为了利用token之间的相对位置信息，假定query向量$q_m$和key向量$k_n$之间的内积操作可以被一个函数g表示，该函数g的输入时词嵌入向量$x_m$，$x_n$和他们之间的相对位置m-n：
$$
\lang f_q(x_m,m),f_k(x_n,n) \rang = g(x_m, x_n, m-n)
$$

接下来的目标就是找到一个等价的位置编码方式，从而使得上述关系成立。
假定现在词嵌入向量的维度$d=2$，利用二维平面上的向量的几何性质，然后提出了一个满足上述关系的f和g的形式如下：
$$
f_q(x_m, m) = (W_qx_m)e^{im \theta}  \\
f_k(x_n, n) = (W_kx_n)e^{im \theta}  \\
g(x_m, x_n, m-n) = R_e[(W_qx_m)(W_kx_n) * e^{i(m-n)\theta}]
$$

其中，$R_e$表示复数的实部。
进一步，$f_q$可以表示为：
$$
f_q(x_m, m) = 
\begin{pmatrix}
\cos m \theta & -sin m \theta  \\
sin m \theta & cos m \theta
\end{pmatrix}
\begin{pmatrix}
W_q^{(1,1)} & W_q^{(1,2)}  \\
W_q^{(2,1)} & W_q^{(2,2)}
\end{pmatrix}
\begin{pmatrix}
x_m^{(1)} \\
x_m^{(2)}
\end{pmatrix}  \\
=
\begin{pmatrix}
\cos m \theta & -\sin m \theta \\
\sin m \theta & \cos m \theta
\end{pmatrix}
\begin{pmatrix}
q_m^{(1)} \\ 
q_m^{(2)}
\end{pmatrix}
$$

看到这就会发现，其实是query向量乘了一个旋转矩阵。
同理，$f_k$可以表示成下下面的式子：
$$
f_k(x_m, m) = 
\begin{pmatrix}
\cos m \theta & -sin m \theta  \\
sin m \theta & cos m \theta
\end{pmatrix}
\begin{pmatrix}
W_k^{(1,1)} & W_k^{(1,2)}  \\
W_k^{(2,1)} & W_k^{(2,2)}
\end{pmatrix}
\begin{pmatrix}
x_m^{(1)} \\
x_m^{(2)}
\end{pmatrix}  \\
=
\begin{pmatrix}
\cos m \theta & -\sin m \theta \\
\sin m \theta & \cos m \theta
\end{pmatrix}
\begin{pmatrix}
k_m^{(1)} \\ 
k_m^{(2)}
\end{pmatrix}
$$

最终$g(x_m,x_n,m-n)$可以表示如下：
$$
g(x_m, x_n, m-n) = (q_m^{(1)} & q_m^{(2)})
$$


# 参考
<https://zhuanlan.zhihu.com/p/647109286>

