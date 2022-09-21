import torch
from  torch import nn
from torch.nn import functional as F

class ResBlk1(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(ResBlk1,self).__init__()
        #if norm_layer ==None:
        #     norm_layer=RepresentativeBatchNorm2d

        self.conv1=nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 =nn.BatchNorm2d(ch_out)
        self.extra=nn.Sequential()
        if ch_out !=ch_in:
            self.extra=nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1),
                nn.BatchNorm2d(ch_out)
            )
    def forward(self,x):
        # [b,ch,h,w]
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        # short out
        out=self.extra(x)+out
        return out
class ResBlk2(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(ResBlk2,self).__init__()
        #if norm_layer ==None:
        #     norm_layer=RepresentativeBatchNorm2d

        self.conv1=nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(ch_out)
        # self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        # self.bn2 =nn.BatchNorm2d(ch_out)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.extra=nn.Sequential()
        if ch_out !=ch_in:
            self.extra=nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=2),
                nn.BatchNorm2d(ch_out)
            )
    def forward(self,x):
        # [b,ch,h,w]
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.maxpool(out)
        # short out
        out=self.extra(x)+out
        return out

class  ResNet18 (nn.Module):
    def __init__(self):
        super (ResNet18,self).__init__()
        # self.conv1=nn.Sequential(
        #     nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
        #     nn.BatchNorm2d(64),
        #     # nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        #     # nn.BatchNorm2d(16)
        # )
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
         #[b,64,h,w]=>[b,128,h,w]
        self.blk1= ResBlk1(64,64)
        self.blk2= ResBlk1(64,64)
        self.blk3= ResBlk2(64,128)
        self.blk4= ResBlk1(128, 128)
        self.blk5= ResBlk2(128,256)
        self.blk6 = ResBlk1(256, 256)
        self.blk7= ResBlk2(256,512)
        self.blk8= ResBlk1(512, 512)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.outlayer=nn.Linear(512,5)

        # self.blk1 = ResBlk(16, 16)
        # self.blk2 =  ResBlk(16,32)
        # self.outlayer=nn.Linear(32*32*32,10)

    def forward(self,x):
        #x=F.relu(self.conv1(x))
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.blk1(x)
        x=self.blk2(x)
        x=self.blk3(x)
        x=self.blk4(x)
        x = self.blk5(x)
        x = self.blk6(x)
        x = self.blk7(x)
        x = self.blk8(x)
        x=self.avgpool(x)
        #print(x.shape)

        x=x.view(x.size(0),-1)
        x=self.outlayer(x)

        return x
def main():

    tmp=torch.randn(2,64,224,224)
    #blk1 = ResBlk1(64, 64)
    blk2=ResBlk2(64,128)
    #out=blk1(tmp)
    out=blk2(tmp)
    print(out.shape)

    modle = ResNet18()
    print(modle)
    tmp = torch.randn(2, 3, 224, 224)
    #tmp = torch.randn(3, 3, 64, 64)
    out=modle(tmp)
    print('resnet18:',out.shape)

if __name__=='__main__':
    main()