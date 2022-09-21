from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.pyplot import MultipleLocator
import numpy as np
myfont = font_manager.FontProperties(fname="C:\Windows\Fonts\simfang.ttf")
plt.rcParams["font.sans-serif"] = "SimHei"  # 修改字体的样式可以解决标题中文显示乱码的问题
plt.rcParams["axes.unicode_minus"] = False  # 该项可以解决绘图中的坐标轴负数无法显示的问题


#file_r = open('kaggleminispvdata6//200-minispvdata6-bzresnet18.txt')
file_r = open('kpvdata6//200origin-lr=10-pvdata6-resnet18.txt')
epoch_list=[]
accuracy_list=[]
loss_list=[]

for f in file_r:
    m=f.split(' ')
    loss_list.append(float(m[4]))
    accuracy_list.append(float(m[7])*100)

    e=m[1].split(']')
    #print(e[0])
    epoch_list.append(e[0])
    #print(f.rstrip())
file_r.close()
t=max(accuracy_list)
print(t)


#file_r1 = open('kaggleminispvdata6//200tlfalse-spvdata6-resnet18.txt')
file_r1 = open('kpvdata6//200tlture-lr=5-pvdata6-resnet18.txt')
accuracy_list1=[]
loss_list1=[]

for f in file_r1:
    m=f.split(' ')
    loss_list1.append(float(m[4]))
    accuracy_list1.append(float(m[7])*100)
    #print(m[7])


    #print(f.rstrip())
file_r1.close()
t=max(accuracy_list1)
print(t)
##第三条曲线
#file_r2 = open('kaggleminispvdata6//200tltrue-spvdata6-resnet18-2.txt')
file_r2 = open('kpvdata6//200-spvdata6-googlenet.txt')
accuracy_list2=[]
loss_list2=[]

for f in file_r2:
    m=f.split(' ')
    loss_list2.append(float(m[4]))
    accuracy_list2.append(float(m[7])*100)
    #print(m[7])
    #print(f.rstrip())
file_r2.close()
t=max(accuracy_list2)
print(t)

##第四条曲线
#file_r3 = open('kaggleminispvdata6//200+200tlftlr=5-spvdata6-resnet18.txt')
file_r3 = open('kpvdata6//200-pvdata6-vgg19.txt')
accuracy_list3=[]
loss_list3=[]

for f in file_r3:
    m=f.split(' ')
    loss_list3.append(float(m[4]))
    accuracy_list3.append(float(m[7])*100)
    #print(m[7])
    #print(f.rstrip())
file_r3.close()
t=max(accuracy_list3)
print(t)


x_major_locator = MultipleLocator(19)
# 把x轴的刻度间隔设置为1，并存在变量里
y_major_locator = MultipleLocator(10)
# 把y轴的刻度间隔设置为10，并存在变量里
ax = plt.gca()
# ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
# 把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
# 把y轴的主刻度设置为10的倍数
plt.xlim(-0.5, 200)
# 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
#plt.ylim(-5, 100)
plt.ylim(10.5, 100)
# 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白


from scipy.interpolate import make_interp_spline
plt.title('acc曲线',fontsize=20)
plt.xlabel('epochs',fontsize=20)
plt.ylabel('acc(%)',fontsize=20)
# x_smooth = np.linspace(1, 100, 10)
# y_smooth = make_interp_spline(epoch_list, accuracy_list)(x_smooth)
#plt.plot(x_smooth, y_smooth)
# fit_result_1 = np.polyfit(epoch_list,accuracy_list,1)#进行一次拟合，这个只能返回一个拟合后的系数矩阵，如：[11.256762634722696, -18.173869106185016]，并不能生成图像函数
# fit_ploy_1 = np.poly1d(fit_result_1)

plt.plot(epoch_list, accuracy_list,'-*',label='ResNet18')
plt.plot(epoch_list, accuracy_list1,'--',label='TLResNet18')
plt.plot(epoch_list, accuracy_list2,'-',label='GoogLeNet')
plt.plot(epoch_list, accuracy_list3,'-.',label='VGG19')
#plt.legend(fontsize=20)
plt.legend(loc='lower right', fontsize=20) # 标签位置
#plt.grid()
#plt.axis([0, 30, 0, 1])
plt.show()

x_major_locator = MultipleLocator(19)
# 把x轴的刻度间隔设置为1，并存在变量里
y_major_locator = MultipleLocator(0.2)
# 把y轴的刻度间隔设置为10，并存在变量里
ax = plt.gca()
# ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
# 把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
# 把y轴的主刻度设置为10的倍数
plt.xlim(-0.5,200)
# 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
plt.ylim(0, 3)
# 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白

plt.title('loss曲线',fontsize=20)
plt.xlabel('epochs',fontsize=20)
plt.ylabel('loss',fontsize=20)
ax = plt.gca()

plt.plot(epoch_list, loss_list,'-*',label='ResNet18')
plt.plot(epoch_list, loss_list1,'-.',label='TLResNet18')
plt.plot(epoch_list, loss_list2,'-',label='GoogLeNet')
plt.plot(epoch_list, loss_list3,'-.',label='VGG19')
plt.legend(fontsize=20)
#plt.grid()
plt.show()