import torch
import torch.nn as nn
from  torch.utils.data import DataLoader
from  torchvision import transforms,datasets
from torch.nn import functional as F
import matplotlib.pyplot as plt

DOWNLOAD_MNIST = False
batch_size=64
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ),(0.308, ))
])
loss_list=[]
accuracy_list=[]
epoch_list=[]

#数据地址
path='D:/git/datasets/mnist'

train_data = datasets.MNIST(root=path,train=True,transform=transform, download=DOWNLOAD_MNIST)
test_data = datasets.MNIST(root=path,train=False,transform=transform, download=DOWNLOAD_MNIST)
train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=False)

class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.channels=channels
        self.conv1=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
    def forward(self,x):
        y=F.relu(self.conv1(x))
        y=self.conv2(y)
        return F.relu(x+y)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,16,kernel_size=5)
        self.conv2=nn.Conv2d(16,32,kernel_size=5)
        self.mp=nn.MaxPool2d(2)
        self.reblock1=ResidualBlock(16)
        self.reblock2=ResidualBlock(32)
        self.fc=nn.Linear(32*4*4,10)
    def forward(self,x):
        in_size=x.size(0)
        x=self.mp(F.relu(self.conv1(x)))#[1,28,28]=>[16,24,24]=>[16,12,12]
        x=self.reblock1(x)              #
        x=self.mp(F.relu(self.conv2(x)))#=>[32,8,8]=>[32,4,4]
        x=self.reblock2(x)              #
        x=x.view(in_size,-1)
        return self.fc(x)

model=Net()
#x=torch.randn([28,28])
#y=model(x)

#device=torch.device('cuda')
#model.to(device)
print(model)

criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    runing_loss=0
    for batch_index,data in enumerate(train_loader,0):
        inputs,target=data
        #inputs,target=input.to(device)
        optimizer.zero_grad()

        outputs=model(inputs)
        loss=criterion(outputs,target)
        loss.backward()
        optimizer.step()

        runing_loss+=loss.item()

        if batch_index % 200==199:
            print('[%d,%d]loss:%.3f'%(epoch+1,batch_index+1,runing_loss/300))
            if batch_index+1==200:
                loss_list.append(runing_loss/300)
                print(loss_list)
            runing_loss=0.0

def run_test_sets():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            #images,labels=images.to(device), labels.to(device)
            outputs=model(images)
            _,predicts=torch.max(outputs.data,dim=1)
            total+=labels.size(0)
            correct+=(predicts==labels).sum().item()
    print('accuracy on test sets: %d%%'%(100*correct/total))
    print('accuracy/total:[',correct,'/',total,']')
    accuracy_list.append(correct/total)
    #print(accuracy_list)
if __name__=='__main__':


    for epoch in range(3):
        train(epoch)
        epoch_list.append(epoch)
        run_test_sets()
    torch.save(model,'net.pth')
    torch.save(model.state_dict(), 'net_params.pth')
    plt.plot(epoch_list,accuracy_list)
    plt.grid()
    plt.show()
    plt.plot(epoch_list,loss_list)
    plt.grid()
    plt.show()