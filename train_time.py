
file_r = open('spv_resnet//kaggle//200spv-resnet//200-spvdata-googlenet.txt')
epoch_list=[]
accuracy_list=[]
loss_list=[]

sum=0
for f in file_r:
    m=f.split(' ')


    print(m[10])
    sum=sum+float(m[10])
    avg=sum/200
    #print(f.rstrip())
file_r.close()
print(sum)
print(avg)