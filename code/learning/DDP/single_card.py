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
    data_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=1024, shuffle=True)

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