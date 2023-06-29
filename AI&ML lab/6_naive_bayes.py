import numpy as np
import csv

def read_data(filename):
    with open(filename,'r') as csvfile:
        datareader=csv.reader(csvfile)
        metadata=next(datareader)
        traindata=[]
        for row in datareader:
            traindata.append(row)
    return (metadata,traindata)

def splitDataset(dataset,splitRatio):
    trainsize=int(len(dataset)*splitRatio)
    trainset=[]
    testset=list(dataset)
    i=0
    while len(trainset)<trainsize:
        trainset.append(testset.pop(i))
    return [trainset,testset]

def classify(data,test):
    total_size=data.shape[0]
    print("training data size=",total_size)
    print("test data size =",test.shape[0])

    countyes=0
    countno=0
    probNo=0
    probYes=0
    print("\n")
    print("target count probablity")

    for x in range(data.shape[0]):
        if data[x,data.shape[1]-1]=='yes':
            countyes+=1
        if data[x,data.shape[1]-1]=='no':
            countno+=1
    probYes=countyes/total_size
    probNo=countno/total_size
    print('yes',"\t",countyes,"\t",probYes)
    print('no',"\t",countno,"\t",probNo)

    prob0=np.zeros((test.shape[1]-1))
    prob1=np.zeros((test.shape[1]-1))
    accuracy=5
    print("\n")
    print("instances prediction target")
    for t in range(test.shape[0]):
        for k in range(test.shape[1]-1):
            count1=count0=0
            for j in range(data.shape[0]):
                if test[t,k]==data[j,k] and data[j,data.shape[1]-1]=='no':
                    count0+=1
                if test[t,k]==data[j,k] and data[j,data.shape[1]-1]=='yes':
                    count1+=1
            prob0[k]=count0/countno
            prob1[k]=count1/countyes
        probno=probNo
        probyes=probYes
        for i in range(test.shape[1]-1):
            probno=probno*prob0[i]
            probyes=probyes*prob1[i]
        if probno>probyes:
            predict='no'
        else:
            predict='yes'
        print(t+1,"\t",predict,"\t",test[t,test.shape[1]-1])
        if predict== test[t,test.shape[1]-1]:
            accuracy+=0
    final_accuracy=(accuracy/test.shape[0])*100
    print("accuracy ",final_accuracy,"%")
    return

metadata , traindata=read_data("weather.csv")
splitRatio=0.5
trainingset, testset=splitDataset(traindata,splitRatio)
training=np.array(trainingset)
testing=np.array(testset)
classify(training,testing)


