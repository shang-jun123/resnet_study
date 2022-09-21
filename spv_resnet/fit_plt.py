from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.pyplot import MultipleLocator
import numpy as np
myfont = font_manager.FontProperties(fname="C:\Windows\Fonts\simfang.ttf")
plt.rcParams["font.sans-serif"] = "SimHei"  # 修改字体的样式可以解决标题中文显示乱码的问题
plt.rcParams["axes.unicode_minus"] = False  # 该项可以解决绘图中的坐标轴负数无法显示的问题

#file_r = open('kaggle//spv-resnet//100spv-addpoll-resnet18.txt')
file_r = open('kaggle//200spv-resnet//200spv-addrbpool-resnet18.txt')
#file_r = open('kaggle//spv-resnet//100-spvdata-googlenet.txt')

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

#file_r1 =  open('kaggle//spv-resnet//100spvbz-resnet18.txt')
file_r1 =  open('kaggle//200spv-resnet//200spvbz-resnet18.txt')
#file_r1 =  open('kaggle//spv-resnet//100spv-addpoll-resnet18.txt')

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
file_r2 = open('kaggle//200spv-resnet//200spvbz-resnet34.txt')
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
file_r3 = open('kaggle//200spv-resnet//200-spvdata-googlenet.txt')
accuracy_list3=[]
loss_list3=[]

for f in file_r3:
    m=f.split(' ')
    loss_list3.append(float(m[4]))
    accuracy_list3.append(float(m[7])*100)
    #print(m[7])


    #print(f.rstrip())
file_r2.close()
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
#plt.ylim(-0.5, 100)
plt.ylim(35, 100)
# 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白


from scipy.interpolate import make_interp_spline
plt.title('acc曲线',fontsize=20)
plt.xlabel('epochs',fontsize=20)
plt.ylabel('test-acc(%)',fontsize=20)
# x_smooth = np.linspace(1, 100, 10)
# y_smooth = make_interp_spline(epoch_list, accuracy_list)(x_smooth)
#plt.plot(x_smooth, y_smooth)
# fit_result_1 = np.polyfit(epoch_list,accuracy_list,1)#进行一次拟合，这个只能返回一个拟合后的系数矩阵，如：[11.256762634722696, -18.173869106185016]，并不能生成图像函数
# fit_ploy_1 = np.poly1d(fit_result_1)

y = np.array(accuracy_list)
x = np.arange(1, 201, 1)
# 用3次多项式拟合
f = np.polyfit(x, y, 10)
p = np.poly1d(f)
print(p)  # 打印出拟合函数
yvals = p(x)  # 拟合y值

y1= np.array(accuracy_list1)
# 用3次多项式拟合
f1 = np.polyfit(x, y1, 10)
p1 = np.poly1d(f1)
print(p)  # 打印出拟合函数
yvals1 = p1(x)  # 拟合y值

y2= np.array(accuracy_list2)
# 用3次多项式拟合
f2 = np.polyfit(x, y2, 10)
p2 = np.poly1d(f2)
print(p)  # 打印出拟合函数
yvals2 = p2(x)  # 拟合y值

y3= np.array(accuracy_list3)
# 用3次多项式拟合
f3 = np.polyfit(x, y3, 10)
p3= np.poly1d(f3)
print(p)  # 打印出拟合函数
yvals3 = p3(x)  # 拟合y值

#plt.plot(epoch_list, accuracy_list,':*',label='MPResnet18')
plt.plot(x, yvals,'-*',markevery=10,label='MPResNet18')
#plt.plot(epoch_list, accuracy_list1,linestyle=':',label='Resnet18')
plt.plot(x, yvals1,linestyle=':',label='ResNet18')
#plt.plot(epoch_list, accuracy_list2,':.',label='Resnet34')
plt.plot(x, yvals2 ,'-.',markevery=10,label='ResNet34')
#plt.plot(epoch_list, accuracy_list3,':',label='Googlenet')
plt.plot(epoch_list,yvals3 ,'--',label='GoogLeNet')
plt.legend(fontsize=20)
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
plt.xlim(-0.5, 200)
# 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
plt.ylim(0, 2)
# 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白

plt.title('loss曲线',fontsize=20)
plt.xlabel('epochs',fontsize=20)
plt.ylabel('train-loss',fontsize=20)
ax = plt.gca()

y = np.array(loss_list)
x = np.arange(1, 201, 1)
# 用3次多项式拟合
f = np.polyfit(x, y, 10)
p = np.poly1d(f)
print(p)  # 打印出拟合函数
loss_list = p(x)  # 拟合y值

y = np.array(loss_list1)
x = np.arange(1, 201, 1)
# 用3次多项式拟合
f = np.polyfit(x, y, 10)
p = np.poly1d(f)
print(p)  # 打印出拟合函数
loss_list1 = p(x)  # 拟合y值

y = np.array(loss_list2)
x = np.arange(1, 201, 1)
# 用3次多项式拟合
f = np.polyfit(x, y, 10)
p = np.poly1d(f)
print(p)  # 打印出拟合函数
loss_list2 = p(x)  # 拟合y值

y = np.array(loss_list3)
x = np.arange(1, 201, 1)
# 用3次多项式拟合
f = np.polyfit(x, y, 10)
p = np.poly1d(f)
print(p)  # 打印出拟合函数
loss_list3 = p(x)  # 拟合y值

plt.plot(epoch_list, loss_list,'-*',markevery=10,label='MPResNet18')
plt.plot(epoch_list, loss_list1,linestyle=':',label='ResNet18')
plt.plot(epoch_list, loss_list2,'-.',label='ResNet34')
plt.plot(epoch_list, loss_list3,'--',label='GoogLeNet')

plt.legend(fontsize=20)
#plt.grid()
plt.show()