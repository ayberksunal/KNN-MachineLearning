import pandas as pd
import numpy as np
from math import *
# author: Ayberk SUNAL
# Machine Learning Mini Project


#prepares features and label
#and send it to splitdata() to seperate rows two parts
def helper():
    df = pd.read_csv('data.csv')
    dfFeatures = df.loc[:, ['height', 'midtermGrade', 'hoursOfStudy']]
    dfLabel = df.loc[:, ['label']]
    percentage = 0.7

    print(type(dfFeatures))

    global features_Test,features_Train,label_Train,label_Test
    features_Train, features_Test = splitData(dfFeatures, percentage)
    label_Train, label_Test = splitData(dfLabel, percentage)


#seperates the data to two parts that wanted
def splitData(df,r):
	train=df.iloc[:int(r*100),:]#start from 71 until 100
	test=df.iloc[int(r*100):,:]#example: first 70 samples.
	return train,test

helper()

#data frame to numpy array for euclid distance calculation
numpyMatrix_Features_Train = features_Train.as_matrix()
numpyMatrix_Features_Test =  features_Test.as_matrix()
numpyMatrix_Label_Train = label_Train.as_matrix()
numpyMatrix_Label_Test = label_Test.as_matrix()


shp=(len(features_Test),len(features_Train))
euclid_zero_train = np.zeros(shp)
print(euclid_zero_train.shape)

#print('numpyMatrix_Features_Train',numpyMatrix_Features_Train)
#print('numpyMatrix_Features_Test',numpyMatrix_Features_Test)


#calcualtes euclidian distance of 2 points
def euclidDist2(x,y):
    distances = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
    print(type(distances))
    return distances


#calcualtes euclidian distance of 2 points
def euclidDist(x,y):
    for a in range(len(x)):
        for b in range(len(numpyMatrix_Features_Train)):
            result = np.sqrt((x[a,0] - y[b,0]) ** 2 +(x[a,1] - y[b,1]) ** 2 +(x[a,2] - y[b,2]) ** 2)
            euclid_zero_train[a,b] = result
    print(type(euclid_zero_train))
    return euclid_zero_train

#calculates the knn and finds the smallest euclidian value
#selects the most frequent label value for one prediction and lists them
Y_predict= []
Y_predict_most_freq= []
def predict(dist,k,Ytrain):
    sorted_euclid = np.argsort(dist)
   # print(sorted_euclid)
    k_array = np.arange(k)
    cutted_sorted_euclid = sorted_euclid[:,k_array]
    #print('cutted',cutted_sorted_euclid)
    for i in cutted_sorted_euclid:
        Y_predict_one_line = []
        num_pass = 0
        num_fail = 0
        for j in i:
            label_future = Ytrain[j].astype(str)
            Y_predict_one_line.append(label_future)
            #counts the values of one point
            if label_future == "fail":
                num_fail += 1
            elif label_future =="pass":
                num_pass += 1
        print('one', Y_predict_one_line)
        #adds the most freq. value(selects)
        if num_fail > num_pass:
            Y_predict_most_freq.append("fail")
        else:
            Y_predict_most_freq.append("pass")
        Y_predict.append((Y_predict_one_line))
    return Y_predict_most_freq


#compares the predicted and known label result to see the quality of the result
def compare(predicted_label,test_label):
    true_label=0
    false_label=0
    print('predicted_label',predicted_label)
    print('test_label', test_label)
    print('len(predicted_label)',len(predicted_label))
    print('len(test_label)',len(test_label))
    #counts the true false labels
    for i in range(len(predicted_label)):
        if predicted_label[i]==test_label[i]:
            true_label +=1
        else:
            false_label +=1
    #calculates the percantage of quality
    accuracy = (true_label*100)/len(test_label)
    return true_label,accuracy






#euclidDist(numpyMatrix_Features_Test,numpyMatrix_Features_Train)
euclidDist2(numpyMatrix_Features_Test,numpyMatrix_Features_Train)
# print(predict(euclid_zero_train,3,numpyMatrix_Label_Train))
# print(numpyMatrix_Label_Test.tolist())
# print(label_Test['label'].tolist())

print('######## Features Train #####')
print(features_Train)
print('######## Features Test #####')
print(features_Test)

print('######## Label Train #####')
print(label_Train)
print('######## Label Test #####')
print(label_Test)

#print(euclid_zero_train)

euclid_result = euclidDist2(numpyMatrix_Features_Test,numpyMatrix_Features_Train)
np.savetxt('file.txt',euclid_result)
predicted_result = predict(euclid_result,3,numpyMatrix_Label_Train)
true_prediction,accuracy_prediction = compare(predicted_result,label_Test['label'].tolist())
print('True Prediction = {} Accuracy = {}%'.format(true_prediction,accuracy_prediction))
