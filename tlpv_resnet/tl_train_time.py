
file_r = open('30-spvdata//30-spvdata_resnet50.txt')
epoch_list=[]
accuracy_list=[]
loss_list=[]

sum=0
for f in file_r:
    m=f.split(' ')


    print(m[10])
    sum=sum+float(m[10])
    avg=sum/30
    min=sum/60
    #print(f.rstrip())
file_r.close()
print(sum)
print(avg)
print(min)