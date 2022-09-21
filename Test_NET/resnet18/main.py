import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torch import nn
from lenet import Lenet
from resnet18 import ResNet18
import  matplotlib.pyplot  as plt
import time

#数据集为cifar10
def main():
    batch_size=32
    download=False
    path='D:/git/datasets/cifar'
    transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
    cifar_train=datasets.CIFAR10(path,
                                 True,
                                 transform=transform,
                                 download=download)
    cifar_train=DataLoader(cifar_train,batch_size=batch_size,shuffle=True)
    cifar_test = datasets.CIFAR10(path, False, transform=transform, download=download)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)

    x,label=iter(cifar_train).next()
    print('x',x.shape,'label',label.shape)
    criteon=nn.CrossEntropyLoss()
    #model=Lenet()
    model=ResNet18()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

    print(model)
    model.train()
    loss_list = []
    accuracy_list = []
    epoch_list = []
    for epoch in range(10):
        t1=time.perf_counter()
        for batch_index, (x,label) in enumerate(cifar_train):
            logits=model(x)
            # logits [b,10]
            # label  [b]
            loss=criteon(logits,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch:',epoch, 'loss:%.3f'%loss.item())

        # test
        model.eval()
        with torch.no_grad():
            total_correct=0
            total_num=0
            for x, label in cifar_test:
                # [b,10]
                logits=model(x)
                # [b]
                pred=logits.argmax(dim=1)
                #  scalar tensor
                total_correct+=torch.eq(pred,label).float().sum().item()
                total_num+=x.size(0)

            acc=total_correct/total_num
            epoch_list.append(epoch + 1)
            accuracy_list.append(acc* 100)
            print('acc:{:.2%}  train_time:{:.2f}s'.format(acc,(time.perf_counter()-t1)))

    print('Finished Training')
    plt.rcParams["font.sans-serif"] = "SimHei"  # 修改字体的样式可以解决标题中文显示乱码的问题
    plt.title('准确率曲线')
    plt.xlabel('epochs')
    plt.ylabel('acc(%)')
    plt.plot(epoch_list, accuracy_list)
    plt.grid()
    plt.show()

    torch.save(model, 'net.pkl')
if __name__=='__main__':
    main()
