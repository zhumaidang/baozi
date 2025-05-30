# MoE（Mixture of Experts）

## 一、概念

> 混合专家模型（MoE）提出的核心思路是将多个领域的复杂问题分配给各领域专家最后汇总结论。每一个专家都在其擅长的领域内做出贡献，门控网络机制决定哪个专家参与解答特定问题

2017年QuocLe等人提出了一种MoE层，通过引入**稀疏性**来大幅提高模型的规模和效率。

核心思想：

MoE的基本理念是将输入数据根据任务类型分割成多个区域，并将每个区域的数据分配一个或多个专家模型。每个专家模型可以专注于处理输入这部分数据，从而提高模型的整体性能。

### 1.1 MoE与集成模型的区别：

|          | MoE                                                          | Ensemble Learning                                            |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 应用目的 | 提高模型收敛和推理速度                                       | 提高模型预测精确度                                           |
| 训练步骤 | 1. 将预测建模任务分为若干个子任务，在每个子任务上训练一个专家模型；<br />2. 开发一个门控模型来预测输入应分配给哪个专家 | 训练多个基学习器，根据bagging、Boosting、Stacking策略输出结果 |

### 1.2 MoE发展

- 2017年，谷歌首次将MoE引入自然语言处理领域，通过在LSTM层之间增加MoE实现了机器翻译方面的性能提升
- 2020年，Gshard首次将MoE技术引入Transfomrer架构，并提供了高效的分布式并行计算架构
- 2021年，V-MoE将MoE架构应用在计算机视觉领域的Transformer架构模型
- 2022年，LIMoE是首个应用了稀疏混合专家模型技术的多模态模型

## 二、关键技术

MoE主要包括两个核心组件：门控模型（GateNet）和专家模型（Experts）。

### 2.1 门控模型

混合专家模型中的“门”是一种稀疏门网络，它接收三个数据元素作为输入，然后输出一个权重，这些权重表示每个专家模型对处理输入数据的贡献。一般是通过softmax门控函数进行建模，并选择前K个。
$$
G(x)=Softmax(KeepTopK(H(x), k)) \\
H(x)_{i}=(x \cdot W_g)_i+StandardNormal()\cdot Softplus((x\cdot W_{noise})_i) \\
KeepTopK(v, k)_i=
\begin{cases}
v_i & \text{if}v_i\text{is in the top}k\text{elements of}v.  \\
-\infty & otherwise.
\end{cases}
$$

### 2.2 专家模型

训练的过程中，输入的数据被门控模型分配到不同的专家模型中进行处理；在推理过程中，被门控选择的专家会针对输入的数据，产生相应的输出。这些输出最后会和每个专家模型处理该特征的能力分配的权重进行加权组合，形成最终的预测结果。
$$
Importance(X)=\sum_{x\in X} G(x) \\
L_{importance}(X)=w_{importance}\cdot CV(Importance(X))^2
$$

### 2.3 MoE优势

在传统的密集模型中，每个输入都必须经历完整的计算流程，这导致了在处理大规模数据时的显著计算成本，稀疏混合专家模型只激活部分模型，形成了“稀疏”结构。这种稀疏性被认为是混合专家模型的重要优点，不仅在减少计算负担的同时，还能提高模型的效率和性能。

为了有效的控制稀疏性，主要依赖于门控网络的设计和参数调整。门控网络负责决定哪些专家模型参与处理当前的输入数据。然而，在进行参数选择时需要注意一个**均衡**：如果门控网络在单次选择中激活了较多的专家模型，可能提升模型的表现能力，但却会导致稀疏性的降低，带来额外的计算复杂性和耗时。

因此，需要根据具体的应用需求和计算资源限制来调整门控网络的设计和参数。

### 2.4 实现步骤

具体实现步骤：

1. 前向传播：输入数据进入混合专家模型，首先进行前向传播。数据同时传递到门控网络，准备进行后续的计算。这一步时信息流的起点，让模型感知输入的特征并为后续步骤做好准备。
2. 门控计算：门控网络接收输入数据并执行一系列学习的非线性变换。这一过程产生了一组权重，这些权重表示了每个专家对当前输入的贡献程度。通常，这些权重经过softmax等函数的处理，以确保他们相加为1，形成了一个概率分布。这样的分布表示了在给定输入情境下每个专家被激活的概率。

![img](https://picx.zhimg.com/v2-8a947aabe9563ce8f67f036452a840a1_r.jpg)

3. 专家模型：数据经过门控网络选择后进入每个专家模型，每个专家根据其设计和参数对输入进行处理。专家模型可以视为是对输入数据的不同方面或特征进行建模的子模型。**每个专家产生的输出是对输入数据的一种表示**，这些表示将在后续的步骤中进行加权聚合。
3. 加权聚合：专家模型的输出由门控网络计算的权重进行加权聚合。每个专家的输出乘以其相应的权重，并将这些加权的输出求和，形成最终的模型输出。这种加权的组合机制使得模型能够在不同输入下自适应地选择哪个专家模型地输出对当当前任务更具有利。
3. 反向传播和更新：模型地训练在这一阶段通过反向传播算法进行。损失函数地梯度用于调整门控网络和专家模型的参数，以最小化预测值与实际标签之间的误差。这一过程是训练模型权重的关键步骤，确保模型能够更好地适应训练数据。
3. 稀疏性调整：通过引入适当的正则化项，可以调整模型的稀疏性。正则化项在门控网络的损失函数中起到作用，控制专家模型的激活状态，从而影响模型的整体稀疏性。这是一个需要仔细平衡的参数，以满足对模型效率和性能之间的不同需求。
3. 动态适应性：由于门控网络的存在，混合专家模型能够实现动态适应性。根据输入数据的不同，模型可以自动调整专家模型的使用，从而更灵活地适应不同的输入分布和任务场景。



## 三、混合专家模型的问题思考：通信权衡

**MOE优势**

1. 任务特异性：采用混合专家方法可以有效地充分利用多个专家模型地优势，每个专家都可以专门处理不同的任务或数据的不同部分，在处理复杂任务时取得更卓越的性能。各个专家模型能够针对不同的数据分布和模式进行建模，从而显著提升模型的准确性和泛化能力，因此模型可以更好地使用任务地复杂性，这种任务特异性使得混合专家模型在处理多模态数据和复杂任务时表现出色。
2. 灵活性：混合专家方法展现出卓越地灵活性，能够根据任务的需求灵活选择并组合适宜的专家模型。模型的结构允许根据任务的需要动态选择激活的专家模型，实现对输入数据的灵活处理。
3. 高效性：由于只有少数专家模型被激活，大部分模型处理未激活状态，混合专家模型具有很高的稀疏性。这种稀疏性带来了计算效率的提升，因为只有特定的专家模型对当前输入进行处理，减少了计算的开销
4. 表现能力：每个专家模型可以被设计为更加专业化，能够更好地捕捉输入数据中的模式和关系。整体模型通过组合这些专家的输出，提高了对复杂数据结构的建模能力，从而增强了模型的性能。
5. 可解释性：由于每个专家模型相对独立，因此模型的决策过程更易于解释和理解，为用户提供更高的可解释性，这对于一些对模型决策过程有强解释要求的应用场景非常重要。
6. 适应大规模数据：混合专家方法是处理大规模数据集的理想选择，能够有效地应对数据量巨大和特征复杂的挑战，可以利用稀疏矩阵的高效计算，利用GPU的并行能力计算所有专家层，能够有效地应对海量数据和复杂特征的挑战。其并行处理不同子任务的特性，充分发挥计算资源，帮助有效地扩充模型并减少训练时间，提高模型在训练和推理阶段地效率，使其在大规模数据下具有较强地可扩展性，以更低的计算成本获得更好的结果。这种优势使得混合专家方法成为在大数据环境下进行深度学习的强有力工具。

**面临的问题**

1. 训练复杂性：混合专家模型的训练相对复杂，尤其是涉及到门控网络的参数调整，为了正确地学习专家地权重和整体模型地参数，可能需要更多的训练时间。
2. 超参数调整：选择适当的超参数，特别是与门控网络相关的参数，以达到最佳性能，是一个复杂的任务，这可能需要通过交叉验证等技术进行仔细调整。
3. 专家模型设计：专家模型的设计对模型的性能影响显著，选择适当的专家模型结构，确保其在特定任务上有足够的表现力，是一个挑战。
4. 稀疏性失真：在某些情况下，为了实现稀疏性，门控网络可能会过度地激活i或不激活某些专家，导致模型性能下降，需要谨慎设计稀疏性调整策略，以平衡效率和性能。
5. 动态性问题：在处理动态或快速变化地数据分布时，门控网络可能需要更加灵活的调整，以适应输入数据的变化，这需要额外的处理和设计。
6. 对数据噪声的敏感性：混合专家模型对于数据中的噪声相对敏感，可能在一些情况下表现不如其他更简单的模型



## 四、Tricks

Q：需要将输入路由到不止一个专家

A：便于门控网络学会有效的路由选择，因此至少需要选择两个专家

Q：为什么添加噪声数据

A：为了专家间的负责均衡

Q：如何避免所有的token都被发送到只有少数几个受欢迎的专家

A：引入一个辅助损失，旨在鼓励给予所有专家相同的重要性，这个损失确保所有专家接收到大概相等数量的训练样本，从而

Q：经典工作有哪些？

A：Switch Transformers













# 参考

[大模型的研究新方向：混合专家模型（MoE） - 知乎](https://zhuanlan.zhihu.com/p/672025580)

[一文带你详细了解：大模型MoE架构（含DeepSeek MoE详解） - 知乎](https://zhuanlan.zhihu.com/p/1893017139650216570)