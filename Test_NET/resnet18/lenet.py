
import torch
from torch import nn
from torch.nn import functional as F
class Lenet(nn.Module):
    # for cifar dataset
    def __init__(self):
        super(Lenet,self).__init__()
        self.conv_unit=nn.Sequential(
            # x: [b,3,32,32]=>[b,6, ]
            nn.Conv2d(3,6,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            #
            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
            #
        )
        # faltten
        # fc unit
        self.fc_unit=nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

        # ues Cross Enstory Loss
        #self.criteon=nn.CrossEntropyLoss()


    def forward(self,x):
        batch_size=x.size(0)
        # [b,3,32,32]=>[b,16,5,5]
        x=self.conv_unit(x)
        # [b,16,5,5]=>[b,16*5*5]
        x=x.view(batch_size,16*5*5)
        # [b,16*5*5]=>[b,10]
        logits=self.fc_unit(x)
        # [b,10]
        #pre=F.softmax(logits,dim=1)
        #loss=self.criteon(logits,y)
        return logits

def main():
    net=Lenet()
    # [b,3,32,32]
    tmp = torch.randn(2, 3, 32, 32)
    out =net(tmp)
    # [b,16,5,5]
    print('Lenet out:', out.shape)
    print(net)
if __name__=='__main__':
    main()
