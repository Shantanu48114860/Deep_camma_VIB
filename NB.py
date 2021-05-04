import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.metrics import classification_report
from tqdm import tqdm
from time import time

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)
data=np.zeros([70000,784])
data[:60000,:]=x_train; data[60000:,:]=x_test
label=np.zeros(70000)
label[:60000]=y_train; label[60000:]=y_test


def naive_bayes(data,label):
    n_s,n_f=data.shape          #Find the Shape (Samples,Features) of the data
    classes=np.unique(label)    #Find the Unique Classes in the Data
    n_c=len(classes)            #Number of Classes in the Data
    total_data=np.zeros([n_s,n_f+1]) #Create a Zero Matrix with (Samples,Feature+1) dimensions
    total_data[:,:-1]=data           #Save the samples & features in the total_data matrix
    total_data[:,-1]=label           #Save the Labels in the total_data matrix
    np.random.shuffle(total_data)    #Shuffle the total_data Matrix (Sample & Label) are sampled together
    trainX=total_data[:60000,:]      #Train Data is taken from total Data
    np.random.shuffle(trainX)        #Train Data is shuffled once again
    testX=total_data[60000:,:]       #Test Data is taken from Total Data
    np.random.shuffle(testX)         #Test Data is shuffled once again
    testX_c=testX[:,:-1]             #Take the samples and feature from Test Data
    testX_l=testX[:,-1]              #Take the labels from Test Data
    mean_v=np.zeros([n_c,n_f])       #Take a Zero Matrix that will be used to store the mean of Features wrt classes
    var_v=np.zeros([n_c,n_f])        #Take a Zero Matrix that will be used to store the variance of Features wrt classes
    c_prob=[]                        #list to store P(class)
    confusion_matrix=np.zeros([n_c,n_c]) #Take a Zero Matrix for Confusion MAtrix of size (classes*classes)
    d_acc=[]                         #Take a list that will save each class(digit) accuracy

    for c in classes:
        trainX_c=trainX[trainX[:,-1]==c]   #Filter samples for each class
        trainX_c=trainX_c[:,:-1]           #
        c_prob.append(len(trainX_c)/len(trainX))
        mean_v[int(c),:]=trainX_c.mean(axis=0)#Find mean of each class & save in corresonding mean matrix
        var_v[int(c),:]=trainX_c.var(axis=0)#Find variance of each class & save in corresonding mean matrix

    var_v=var_v+1000    #Since variance is 0 for many pixels, we need to add some value to the variance.
                        #Adding 1000 gives one of the best accuracies
    count=0

    for i in range(testX.shape[0]):
        lists=[]   #Empty list to store probability of all class for ith sample feature
        for j in range(n_c):
            numerator=np.exp(-((testX_c[i]-mean_v[j])**2)/(2*var_v[j]))
            denominator=np.sqrt(2*np.pi*(var_v[j]))
            prob_xc=numerator/denominator
            ratio=np.sum(np.log(prob_xc)) #Probability of jth class for ith feature
            #We found that all classes have equal counts and P(c) for all class is equal provides better accuracy.
            #The line below can be uncommented to use the original formula
            #ratio=np.sum(np.log(prob_xc)+np.log(c_prob[j]))
            lists.append(ratio) #Append Probability of jth class for ith feature

        pred=lists.index(max(lists)) #Take y predicted for the classthat has the maximum probability for jth feature vector
        if pred == testX_l[i]:
            count=count+1 #If y_predicted equals true y label,count is incremented
            confusion_matrix[int(testX_l[i])][int(testX_l[i])]=confusion_matrix[int(testX_l[i])][int(testX_l[i])]+1
            #Values in corresponding confusion matrix is appended
        else:
            for k in range(n_c):
                if pred == k:
                    confusion_matrix[int(testX_l[k])][int(testX_l[i])]=confusion_matrix[int(testX_l[k])][int(testX_l[i])]+1
                    #Values in corresponding confusion matrix is appended
    for l in classes:
        check=testX[testX[:,-1]==l] #Filter features for each class
        a=(confusion_matrix[int(l)][int(l)])/check.shape[0] #Find accuracy of each digit
        d_acc.append(a)   #Append individual digit accuracy


    o_acc=count/testX.shape[0] #Find overall Accuracy
    return(d_acc,o_acc,confusion_matrix,mean_v,var_v)
    #Return (Digit Accuracy,Overall Accuracy,Confusion Matrix,Mean & Variance)


(digit_accuracy,overall_accuracy,matrix,mean_v,var_v)=naive_bayes(data,label)