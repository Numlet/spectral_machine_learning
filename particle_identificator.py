#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:55:59 2017

@author: jesusvergaratemprado
"""

import numpy as np
import sklearn
from collections import OrderedDict
sumary=np.genfromtxt('sumary.csv',delimiter=',')

all_data=np.genfromtxt('all_data.csv',delimiter=',',dtype=str)


class_dict=OrderedDict()
class_dict['C']=1
class_dict['Ca']=2
class_dict['Cl']=3
class_dict['Met']=4
class_dict['Si']=5
class_dict['Oth']=6

number_to_str=OrderedDict()
clasifications=all_data[:,1]
for key in class_dict:
    clasifications[clasifications==key]=class_dict[key]
    number_to_str[class_dict[key]]=key
                 


clasifications=clasifications.astype(int)

training=clasifications[clasifications!=0]
to_predict=clasifications[clasifications==0]

variables=all_data[:,2:].astype(np.float)
variables_extended=np.concatenate((variables,sumary),axis=1)
variables_extended=np.copy(sumary)


data_training=variables_extended[clasifications!=0]
data_to_predict=variables_extended[clasifications==0]


#%%

from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(data_training, training).predict(data_to_predict)

all_pre=[]
for i in range(len(y_pred)):
    all_pre.append([data_to_predict[i,0],number_to_str[y_pred[i]]])
    
print all_pre


#%%
import numpy as np
from sklearn.decomposition import PCA

X = data_training
pca = PCA(n_components=5)
pca.fit(X)
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
print(pca.explained_variance_ratio_) 


#%%



X = data_training
y = training
from sklearn.svm import SVC
clf = SVC()
clf.fit(X, y) 
predictions=clf.predict(data_to_predict)


for prediction in predictions:
    print number_to_str[prediction]

#%%
X = data_training
y = training
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y) 
#KNeighborsClassifier(...)
predictions=(neigh.predict(data_to_predict))

all_pre=[]
for i in range(len(predictions)):
    all_pre.append([data_to_predict[i,0],number_to_str[predictions[i]]])

print all_pre


