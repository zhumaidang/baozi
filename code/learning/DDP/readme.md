
# DDP(DistributedDataParallel)
DDP是PyTorch提供的一种分布式训练方法，它可以在多个GPU上进行数据并行训练。DDP的基本原理是将模型和数据分成多个部分，每个GPU上训练一部分数据，然后将梯度汇总到一个GPU上进行参数更新。DDP的优点是可以在多个GPU上进行训练，从而提高训练速度。

## 问题
1. 单卡到多卡的过程中，代码修改
2. 参数rank、local_rank、node、gpu的含义及关系
3. DDP的启动方式
4. checkpoint的保存和加载
5. 单卡到分布式提速方式

## 一、基本使用
### 1.1 采用一个简单的MINIST分类例子来分析，单卡到多卡的实现
单卡版本：
```
import torch
import torchvision
import torch.utils.data.distributed
from torchvision import transforms

def main():
    # 数据加载部分，直接利用torchvision中的datasets
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))
    ])
    data_set = torchvision.datasets.MNIST(root='./data', train=True, transform=trans, download=True)
    data_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=32, shuffle=True)

    # 搭建网络
    net = torchvision.models.resnet101(num_classes=10)
    net.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net = net.cuda()

    # 定义loss与opt
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)

    # 网络训练
    for epoch in range(10):
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 10 == 0:
                print('epoch: {}, step: {}, loss: {}'.format(epoch, i, loss.item()))
    # 保存checkpoint
    torch.save(net.state_dict(), 'checkpoint.pth')

if __name__ == '__main__':
    main()

```
多卡分布式：
```
import os 
import torch
import torch.distributed as dist
import torch.utils.data.distributed
from torchvision import transforms
from torch.multiprocessing import Process
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

def main(rank):
    dist.init_process_group(backend='gloo', rank=rank, world_size=3)
    torch.cuda.set_device(rank)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))
    ])
    data_set = torchvision.datasets.MNIST(root='./data', train=True, transform=trans, download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
    data_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=32, shuffle=True， sampler=train_sampler)

    net = torchvision.models.resnet101(num_classes=10)
    net.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net = net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(10):
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 10 == 0:
                print('epoch: {}, step: {}, loss: {}'.format(epoch, i, loss.item()))
    if rank == 0:
        torch.save(net.state_dict(), 'checkpoint.pth')

if __name__ == '__main__':
    processes = []
    for rank in range(3):
        p = Process(target=main, args=(rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

```
主要变动
- 分布式采用多进程模式，正向传播时对数据的分配进行调整，所以dataloader多了一个sampler参数
- 反向传播计算后需要对参数进行共享通信，多了一个allreduce操作
- 保存checkpoint时只保存一个进程的参数

## 二、启动方式
DDP的启动方式形式上有多种，内容上是统一的：都是启动多进程来完成运算。
### 2.1 单机多卡
单机多卡可以直接用Process启动，也可以用torch.multiprocessing.spawn启动,还可以用torchrun启动。这里主要介绍前两种，前面MINIST例子中用的是Process格式，截取启动的位置：
```
# Process格式
if __name__ == '__main__':
    world_size = 3
    processes = []
    for rank in range(world_size):
        p = Process(target=main, args=(rank,world_size))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
```
Spawn本质上就是简化了Process的书写，格式如下：
```
# spawn格式：
def main():
    world_size = 3
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
```
> 注意：
spawn要放入if __name__ == "__main__"中，否则会报错。

Spawn完整示例：
```
import os 
import torch
improt torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

def axample(rank, world_size):
    # 初始化
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    # 创建模型
    model = nn.Linear(10, 10).to(rank)

    # 放入ddp
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    # 训练
    inputs = torch.randn(20, 10).to(rank)
    labels = torch.randn(20, 10).to(rank)
    optimizer.zero_grad()
    outputs = ddp_model(inputs)
    loss_fn(outputs, labels).backward()
    optimizer.step()

def main():
    world_size = 2
    mp.spawn(axample, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
```

### 2.2 多机分布式
多机的启动方式可以是直接传递参数并在代码内部解析环境变量，或者能通过torch.distributed.launch启动，两者在格式上有一定的区别，总之要保证代码与启动方式对应。
#### 2.2.1 方式一：每个进程占用一张卡
注意：
1. dist.init_process_group()中，rank参数为进程的编号，world_size参数为进程的数量。rank需要根据node以及GPU的数量计算。
2. world_size的大小=节点数×GPU数
3. ddp里面的device_ids参数为GPU的编号，需要根据rank计算。

示例：
```demo.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--world_size', type=int)
parser.add_argument('--node_rank', type=int)
parser.add_argument('--master_addr', default="", type=str)
parser.add_argument('--master_port', default="", type=str)
args = parser.parse_args()

def example(local_rank, node_rank, local_size, world_size):
    # 初始化
    rank = node_rank * local_size + local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='ncll', init_method="tcp://{}:{}".fromat(args.master_addr, args.master_port) rank=rank, world_size=world_size)
    # 创建模型
    model = nn.Linear(10, 10).to(local_rank)
    # 放入ddp
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    for i in range(100):
        # 训练
        inputs = torch.randn(20, 10).to(local_rank)
        labels = torch.randn(20, 10).to(local_rank)
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss_fn(outputs, labels).backward()
        optimizer.step()

def main():
    local_size = torch.cuda.device_count()
    print(local_size)
    mp.spawn(example, args=(args.node_rank, local_size, args.world_size), nprocs=local_size, join=True)

if __name__ == '__main__':
    main()
```

启动方式：
```
>>> 节点1
>>> python demo.py --world_size=16 --node_rank=0 --master_addr="192.168.0.1" --master_port=22335

>>> 节点2
>>> python demo.py --world_size=16 --node_rank=1 --master_addr="192.168.0.1" --master_port=22335
```

### 2.2.2 方式二：单个进程占用多张卡
注意：
1. dist.init_process_group里面的rank等于节点编号
2. world_size等于节点的总数量
3. DDP不需要指定的device

示例：
```
import torchvision
import torchvision import transforms
import torch.distributed as dist
import torch.utils.data.distributed
import kargparse

parser = argparse.ArgumentParser()
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--master_addr', default="127.0.0.1", type=str)
parser.add_argument('--master_port', default="12355", type=str)
args = parser.parse_args()

def main(rank, world_size):
    dist.init_process_group(backend='gloo', init_method="tcp://{}:{}".format(args.master_addr, args.master_port), rank=rank, world_size=world_size)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))
    ])
    data_set = torchvision.datasets.MNIST(root='./data', train=True, transform=trans, download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
    data_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=32, shuffle=True, sampler=train_sampler, pin_memory=True)

    net = torchvision.models.resnet101(num_classes=10)
    net.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net = net.cuda()

    net = torch.nn.parallel.DistributedDataParallel(net)
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(10):
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 10 == 0:
                print('epoch: {}, step: {}, loss: {}'.format(epoch, i, loss.item()))
if __name__ == '__main__':
    main(args.rank, args.world_size)

```
启动方式：
```
>>> 节点1
>>> python demo.py --world_size=2 --rank=0 --master_addr="192.168.0.1" --master_port=22335
>>> 节点2
>>> python demo.py --world_size=2 --rank=2 --master_addr="192.168.0.1" --master_port=22335
```

### 2.2.3 方式三：torchrun启动
torchrun是pytorch注册的一个用于分布启动的模块，定义在torch/distributed/run.py中。

示例：
```
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        slef.net2 = nn.Sequential(
            nn.Linear(10, 10),
        )
        self.net3 = nn.Linear(10, 10)
        self.layer_norm = nn.LayerNorm(10)

    def forward(self, x):
        retrun self.layer_norm(self.net3(self.net2(self.net1(x))))

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    if rank == 0:
        print(f"local rank: {rank}, world_size: {dist.get_world_size()}")
    torch.cuda.set_device(rank)
    model = DummyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    for i in range(1000):
        inputs = torch.randn(20, 10).to(rank)
        labels = torch.randn(20, 10).to(rank)
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss_fn(outputs, labels).backward()
        optimizer.step()
        if rank == 0 and i % 10 == 0:
            print(f"Iteration:{i/1000*100} %")
    if rank == 0:
        print("Training completed.)

if __name__ == "__main__":
    main()
```
多机启动：
```
>>>节点1
>>> torchrun --nproc_per_node 8 --nnodes 2 --node_rank 0 --rdzv_endpoint 10.192.2.1:62111 ddp_example.py
>>> 节点2
>>> torchrun --nproc_per_node 8 --nnodes 2 --node_rank 1 --rdzv_endpoint 10.192.2.1:62111 ddp_example.py
```
单机启动修改对应参数：
```
torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 61112 ddp_example.py
```
另一种写法：
```
torchrun --nproc_per_node 8 --nnodes=1 --standalone dde_example.py
```

## 三、 参数解读
### 3.1 rank/local_rank/node等的概念
- rank：用于表示进程的编号/序号(在一些结构图中rank指的是软节点，rank可以看成一个计算单位)，每一个进程对应了一个rank的进程，整个分布式由许多rank完成。
- node：物理节点，可以是一台服务器也可以是一个容器，节点内部可以有多个GPU
- rank与local_rank：rank是指整个分布式任务中进程的序号，local_rank是指在一个node上进程的相对序号，local_rank在node之间相互独立
- nnodes/node_rank/nproc_per_node：nnodes是指物理节点数量，node_rank是物理节点的序号，local_rank在node之间相互独立
- world_size：全局中，rank的数量
- group：进程组，一个分布式任务对应了一个进程组，只有用户需要创立多个进程组时才会用到group来管理，默认情况下只有一个group

> 注意：
rank与GPU之间没有必然对应关系，一个rank可以包含多个GPU，一个GPU也可以为多个rank服务

### 3.2 通信参数与模式
通信过程主要是完成模型训练过程中参数信息的传递，主要考虑通信后端和通信模式选择，后端与模式对整个训练的收敛速度影响较大，相差可达2~10倍。DDP中支持了几个常见通信库，而数据处理的模式写在pytorch底层，供用户选择的主要是后端，在初始化时需要设置：
- backend： nccl（NVIDIA），gloo（Facebook），mpi（OpenMPI）。如果显卡支持nccl，建议选择nccl，
- master_add/master_port：主节点的地址与端口，供init_method的tcp方式使用，因为pytorch中网络通信建立是从机去连接主机，运行ddp只需要指定主节点的IP与端口，其他节点的IP不需要填写。这两个参数可以通过环境变量或者init_method传入。

### 3.3 分布式任务中常用的函数
```
torch.distributed.is_ncll_available()  # 判断nccl是否可用
torch.distributed.is_mpi_available()  # 判断mpi是否可用
torch.distributed.is_gloo_available()  # 判断gloo是否可用

# all_reduce操作：将不同rank进程的数据进行操作，比如sum操作。
```

## 四、提速参数与隐藏的简单问题
### 4.1 dataloader提速的参数
- num_worker：加载数据的进程数量，默认只有1个，增加该数量能够提升数据的读入速度。
- pin_memory：锁页内存，加快数据在内存上的传递速度，若数据加载成为训练速度的瓶颈，可以考虑将这两个参数加上。

### 4.2 checkpoint的保存与加载
保存：一般情况下，只需要保存一份ckpt即可，可以用rank来指定一个进程保存：
```
if torch.distributed.get_rank() == 0:
    torch.save(net, "net.pth")
```
加载：加载不同于保存，可以让每个进程独立的加载，也可以让某个rank加载后然后进行传播。注意，当模型大的情况下，独立加载最好将模型映射到cpu上，不然容易出现加载模型的OOM
```
torch.load(model_path, map_localion='cpu')
```

### 4.3 dist.init_process_group的init_method方式
init_method支持tcp和共享文件两种，一般情况下我们使用tcp方式来分享信息，也可以用共享文档，但必须要保证共享文件在每个进程都能访问到，文件系统需要支持锁定。
```
# 方式一：
dist.init_process_group(
    init_method='tcp://10.1.1.20:23456',
    rank=args.rank,
    world_size=4)

# 方式二：
dist.init_process_group(
    init_method='file:///mnt/nfs/sharedfile',
    rank=args.rank,
    world_size=4)
```

### 4.4 进程内指定显卡
目前很多场景下使用分布式都是默认一张卡对应一个进程，所以通常，会设置进程能够看到卡数：
```
# 方式1：在进程内部设置可见的device
torch.cuda.set_device(args.local_rank)
# 方式2：通过ddp里面的device_ids指定
ddp_model = DDP(model, device_ids=[rank]) 
# 方式3：通过在进程内修改环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = loac_rank
```
如果不设置现存可见的参数，那么节点内的rank会调用所用的显卡，这样的话一张显卡可能加载多份模型进行了多份计算，对于大一点的模型或者batch_size设置大的情况下，会导致OOM
对于显存占用小的模型，跑多份的结果有可能提速或者降速，取决于显卡的算力，当一张显卡跑多个模型时，对于算力的压榨方式可以考虑用MPS提速

### 4.5 CUDA初始化的问题

注意：多进程中，防止cuda被初始化多次

## 参考
<https://zhuanlan.zhihu.com/p/358974461>

