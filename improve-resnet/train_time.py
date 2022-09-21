file_r = open('kaggle//200spv-resnet//200spv-addrbpool-resnet18.txt')
epoch_list=[]
accuracy_list=[]
loss_list=[]

sum=0
for f in file_r:
    m=f.split(' ')


    print(m[10])
    sum=sum+float(m[10])

    #print(f.rstrip())
file_r.close()
sum=sum/200
print(sum)