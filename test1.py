tm=[22.8,37.7,65.7,23.5,38.8,64.6,28.7,43.9,65.8,28.9,44.7,66.3]
acc=[91.2,92.4,92.4,92.6,93.1,93.2,93.5,93.9,93.8,93.7,94.3,94.4]
tmin=min(tm)
tmax=max(tm)
# print(tmin)
# print(tmax)
for i in range(12):
    testtm=(tm[i]-tmax)/(tmin-tmax)
    #print(testtm)
    testacc = (acc[i] - min(acc)) / (max(acc)-min(acc))
    #print(testacc)
    ind=testtm+testacc
    print(ind)





