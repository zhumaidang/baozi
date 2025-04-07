import os 
import torch
import torchvision
import torch.distributed as dist
import torch.utils.data.distributed
from torchvision import transforms
from torch.multiprocessing import Process
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'


def main(rank, world_size):
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))
    ])
    data_set = torchvision.datasets.MNIST(root='./data', train=True, transform=trans, download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
    data_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=1024, shuffle=False, sampler=train_sampler)

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
    world_size = 2
    processes = []
    for rank in range(world_size):
        p = Process(target=main, args=(rank,world_size))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()