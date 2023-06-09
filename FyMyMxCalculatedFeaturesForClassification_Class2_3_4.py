# -*- coding: utf-8 -*-
#"""
#Created on Wed May 20 10:52:27 2020

#@author: Bahare Samadi_PhD Thesis, Classification the severity of scoliosis using intervertberal efforts during gait
#"""




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn import svm

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import glob


from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier


# Force Keras to use CPU instead of GPU (no gpu installed, result in error)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

clear = lambda: os.system('cls')  # On Windows System
clear()

print("Import finished")



# Arranging the data
# =============================================================================
#
# =============================================================================
# AIS Class 2 10< CA < 30
# =============================================================================

# = []
# = []
# = []
# = []
# = []
#FyMaxMeanClass2 = []
#MyMaxMeanClass2 = []
#MxMaxMeanClass2 = []
#FyVarMeanClass2 = []
#MyVarMeanClass2 = []
#MxVarMeanClass2 = []
#
#FyMaxMeanClass3 = []
#MyMaxMeanClass3 = []
#MxMaxMeanClass3 = []
#FyVarMeanClass3 = []
#MyVarMeanClass3 = []
#MxVarMeanClass3 = []
#
#FyMaxMeanClass4 = []
#MyMaxMeanClass4 = []
#MxMaxMeanClass4 = []
#FyVarMeanClass4 = []
#MyVarMeanClass4 = []
#MxVarMeanClass4 = []


#directory = "C:/Users/basama/PycharmProjects/Scoliosis Belgium Database/Qact_data/Graphs of 6 cycle per class"
directory = "/Users/baharehsamadi/Google Drive/PhD Final codes_BahareSamadi/Graphs of 6 cycle per class"
for current_file in glob.glob(directory+"/Class2Parameters.csv"):
    Class2parameters = np.genfromtxt(current_file,delimiter=",",skip_header=0,usecols=range(0,26))
    FymeanClass2 = Class2parameters [:, [0]]
    FymeanABSClass2 = Class2parameters [:, [1]]
    FySDClass2 = Class2parameters [:, [2]]
    FyMinClass2 = Class2parameters [:, [3]]
    FyMinClass2ABS = abs(FyMinClass2)
    FyMaxClass2 = Class2parameters [:, [4]]
    FyMaxClass2ABS = abs(FyMaxClass2)
    FyVarClass2 = Class2parameters [:, [5]]
    FyMaxMeanClass2 = Class2parameters [:, [6]]
    FyVarMeanClass2 = Class2parameters [:, [7]]
    
    MymeanClass2 = Class2parameters [:, [8]]
    MymeanABSClass2 = Class2parameters [:, [9]]
    MySDClass2 = Class2parameters [:, [10]]
    MyMinClass2 = Class2parameters [:, [11]]
    MyMinClass2ABS = abs(MyMinClass2)
    MyMaxClass2 = Class2parameters [:, [12]]
    MyMaxClass2ABS = abs(MyMaxClass2)
    MyVarClass2 = Class2parameters [:, [13]]
    MyMaxMeanClass2 = Class2parameters [:, [14]]
    MyVarMeanClass2 = Class2parameters [:, [15]]
    
    MxmeanClass2 = Class2parameters [:, [16]]
    MxmeanABSClass2 = Class2parameters [:, [17]]
    MxSDClass2 = Class2parameters [:, [18]]
    MxMinClass2 = Class2parameters [:, [19]]
    MxMinClass2ABS = abs(MxMinClass2)
    MxMaxClass2 = Class2parameters [:, [20]]
    MxMaxClass2ABS = abs(MxMaxClass2)
    MxVarClass2 = Class2parameters [:, [21]]
    MxMaxMeanClass2 = Class2parameters [:, [22]]
    MxVarMeanClass2 = Class2parameters [:, [23]]
    
    MxdevideMyClass2 = Class2parameters [:, [24]]
    MaxMxdevideMyClass2 = MxdevideMyClass2 [0:8]

    labelClass2 = Class2parameters [:, [25]]


print("Class 2 finished")

directory = "/Users/baharehsamadi/Google Drive/PhD Final codes_BahareSamadi/Graphs of 6 cycle per class"
for current_file in glob.glob(directory+"/Class3Parameters.csv"):
    Class3parameters = np.genfromtxt(current_file,delimiter=",",skip_header=0,usecols=range(0,26))
    FymeanClass3 = Class3parameters [:, [0]]
    FymeanABSClass3 = Class3parameters [:, [1]]
    FySDClass3 = Class3parameters [:, [2]]
    FyMinClass3 = Class3parameters [:, [3]]
    FyMinClass3ABS = abs(FyMinClass3)
    FyMaxClass3 = Class3parameters [:, [4]]
    FyMaxClass3ABS = abs(FyMaxClass3)
    FyVarClass3 = Class3parameters [:, [5]]
    FyMaxMeanClass3 = Class3parameters [:, [6]]
    FyVarMeanClass3 = Class3parameters [:, [7]]
    
    MymeanClass3 = Class3parameters [:, [8]]
    MymeanABSClass3 = Class3parameters [:, [9]]
    MySDClass3 = Class3parameters [:, [10]]
    MyMinClass3 = Class3parameters [:, [11]]
    MyMinClass3ABS = abs(MyMinClass3)
    MyMaxClass3 = Class3parameters [:, [12]]
    MyMaxClass3ABS = abs(MyMaxClass3)
    MyVarClass3 = Class3parameters [:, [13]]
    MyMaxMeanClass3 = Class3parameters [:, [14]]
    MyVarMeanClass3 = Class3parameters [:, [15]]
    
    MxmeanClass3 = Class3parameters [:, [16]]
    MxmeanABSClass3 = Class3parameters [:, [17]]
    MxSDClass3 = Class3parameters [:, [18]]
    MxMinClass3 = Class3parameters [:, [19]]
    MxMinClass3ABS = abs(MxMinClass3)
    MxMaxClass3 = Class3parameters [:, [20]]
    MxMaxClass3ABS = abs(MxMaxClass3)
    MxVarClass3 = Class3parameters [:, [21]]
    MxMaxMeanClass3 = Class3parameters [:, [22]]
    MxVarMeanClass3 = Class3parameters [:, [23]]
    
    MaxMxdevideMyClass3 = Class3parameters [:, [24]]

    labelClass3 = Class3parameters [:, [25]]


print("Class 3 finished")  


directory = "/Users/baharehsamadi/Google Drive/PhD Final codes_BahareSamadi/Graphs of 6 cycle per class"
for current_file in glob.glob(directory+"/Class4Parameters.csv"):
    Class4parameters = np.genfromtxt(current_file,delimiter=",",skip_header=0,usecols=range(0,26))
    FymeanClass4 = Class4parameters [:, [0]]
    FymeanABSClass4 = Class4parameters [:, [1]]
    FySDClass4 = Class4parameters [:, [2]]
    FyMinClass4 = Class4parameters [:, [3]]
    FyMinClass4ABS = abs(FyMinClass4)
    FyMaxClass4 = Class4parameters [:, [4]]
    FyMaxClass4ABS = abs(FyMaxClass4)
    FyVarClass4 = Class4parameters [:, [5]]
    FyMaxMeanClass4 = Class4parameters [:, [6]]
    FyVarMeanClass4 = Class4parameters [:, [7]]
    
    MymeanClass4 = Class4parameters [:, [8]]
    MymeanABSClass4 = Class4parameters [:, [9]]
    MySDClass4 = Class4parameters [:, [10]]
    MyMinClass4 = Class4parameters [:, [11]]
    MyMinClass4ABS = abs(MyMinClass4)
    MyMaxClass4 = Class4parameters [:, [12]]
    MyMaxClass4ABS = abs(MyMaxClass4)
    MyVarClass4 = Class4parameters [:, [13]]
    MyMaxMeanClass4 = Class4parameters [:, [14]]
    MyVarMeanClass4 = Class4parameters [:, [15]]
    
    MxmeanClass4 = Class4parameters [:, [16]]
    MxmeanABSClass4 = Class4parameters [:, [17]]
    MxSDClass4 = Class4parameters [:, [18]]
    MxMinClass4 = Class4parameters [:, [19]]
    MxMinClass4ABS = abs(MxMinClass4)
    MxMaxClass4 = Class4parameters [:, [20]]
    MxMaxClass4ABS = abs(MxMaxClass4)
    MxVarClass4 = Class4parameters [:, [21]]
    MxMaxMeanClass4 = Class4parameters [:, [22]]
    MxVarMeanClass4 = Class4parameters [:, [23]]
    
    MaxMxdevideMyClass4 = Class4parameters [:, [24]]

    labelClass4 = Class4parameters [:, [25]]


print("Class 4 finished")      

####
## Define X, y
###

XFyMaxMean = np.concatenate ((FyMaxMeanClass2 , FyMaxMeanClass3 , FyMaxMeanClass4) ,
                      axis = 0)

XMyMaxMean = np.concatenate ((MyMaxMeanClass2 , MyMaxMeanClass3 , MyMaxMeanClass4),
                      axis = 0)

XMxMaxMean = np.concatenate ((MxMaxMeanClass2 , MxMaxMeanClass3 , MxMaxMeanClass4),
                      axis = 0)

XFyVarMean = np.concatenate ((FyVarMeanClass2 , FyVarMeanClass3 , FyVarMeanClass4) ,
                      axis = 0)

XMyVarMean = np.concatenate ((MyVarMeanClass2 , MyVarMeanClass3 , MyVarMeanClass4),
                      axis = 0)

XMxVarMean = np.concatenate ((MxVarMeanClass2 , MxVarMeanClass3 , MxVarMeanClass4),
                      axis = 0)

XFyMaxABSClass2 = np.max((FyMinClass2ABS,FyMaxClass2ABS),axis = 0)
XMyMaxABSClass2 = np.max((MyMinClass2ABS,MyMaxClass2ABS),axis = 0)
XMxMaxABSClass2 = np.max((MxMinClass2ABS,MxMaxClass2ABS),axis = 0)

XFyMaxABSClass3 = np.max((FyMinClass3ABS,FyMaxClass3ABS),axis = 0)
XMyMaxABSClass3 = np.max((MyMinClass3ABS,MyMaxClass3ABS),axis = 0)
XMxMaxABSClass3 = np.max((MxMinClass3ABS,MxMaxClass3ABS),axis = 0)

XFyMaxABSClass4 = np.max((FyMinClass4ABS,FyMaxClass4ABS),axis = 0)
XMyMaxABSClass4 = np.max((MyMinClass4ABS,MyMaxClass4ABS),axis = 0)
XMxMaxABSClass4 = np.max((MxMinClass4ABS,MxMaxClass4ABS),axis = 0)




XFyDifABSMinMaxClass2 = np.subtract(FyMinClass2ABS,FyMaxClass2ABS)
XMyDifABSMinMaxClass2 = np.subtract(MyMinClass2ABS,MyMaxClass2ABS)
XMxDifABSMinMaxClass2 = np.subtract(MxMinClass2ABS,MxMaxClass2ABS)

XFyDifABSMinMaxClass3 = np.subtract(FyMinClass3ABS,FyMaxClass3ABS)
XMyDifABSMinMaxClass3 = np.subtract(MyMinClass3ABS,MyMaxClass3ABS)
XMxDifABSMinMaxClass3 = np.subtract(MxMinClass3ABS,MxMaxClass3ABS)

XFyDifABSMinMaxClass4 = np.subtract(FyMinClass4ABS,FyMaxClass4ABS)
XMyDifABSMinMaxClass4 = np.subtract(MyMinClass4ABS,MyMaxClass4ABS)
XMxDifABSMinMaxClass4 = np.subtract(MxMinClass4ABS,MxMaxClass4ABS)  

XFyDifABSMinMax = np.concatenate((XFyDifABSMinMaxClass2,XFyDifABSMinMaxClass3,XFyDifABSMinMaxClass4),axis = 0)
XMyDifABSMinMax = np.concatenate((XMyDifABSMinMaxClass2,XMyDifABSMinMaxClass3,XMyDifABSMinMaxClass4),axis = 0)
XMxDifABSMinMax = np.concatenate((XMxDifABSMinMaxClass2,XMxDifABSMinMaxClass3,XMxDifABSMinMaxClass4),axis = 0)                    

XFyMaxABS = np.concatenate((XFyMaxABSClass2,XFyMaxABSClass3,XFyMaxABSClass4),axis = 0)
XMyMaxABS = np.concatenate((XMyMaxABSClass2,XMyMaxABSClass3,XMyMaxABSClass4),axis = 0)
XMxMaxABS = np.concatenate((XMxMaxABSClass2,XMxMaxABSClass3,XMxMaxABSClass4),axis = 0)


XFyABSMean = np.concatenate ((FymeanABSClass2 , FymeanABSClass3 , FymeanABSClass4) ,
                      axis = 0)

XMyABSMean = np.concatenate ((MymeanABSClass2 , MymeanABSClass3 , MymeanABSClass4),
                      axis = 0)

XMxABSMean = np.concatenate ((MxmeanABSClass2 , MxmeanABSClass3 , MxmeanABSClass4),
                      axis = 0)

XFySD = np.concatenate ((FySDClass2 , FySDClass3 , FySDClass4) ,
                      axis = 0)

XMySD = np.concatenate ((MySDClass2 , MySDClass3 , MySDClass4),
                      axis = 0)

XMxSD = np.concatenate ((MxSDClass2 , MxSDClass3 , MxSDClass4),
                      axis = 0)



XFy = np.concatenate((XFyMaxMean,XFyVarMean,XFyMaxABS,XFyABSMean,XFySD), axis = 1)
XMy = np.concatenate((XMyMaxMean,XMyVarMean,XMyMaxABS,XMyABSMean,XMySD), axis = 1)
XMx = np.concatenate((XMxMaxMean,XMxVarMean,XMxMaxABS,XMxABSMean,XMxSD), axis = 1)


X1 = np.concatenate((XFy,XMy,XMx), axis = 1)
# =============================================================================
# Scaling the data
# =============================================================================

scaler = StandardScaler()

X = scaler.fit_transform(X1)
y = np.concatenate ((labelClass2,labelClass3,labelClass4), axis = 0)
y = y.ravel()

print("Define X, y finished") 


from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.preprocessing import MinMaxScaler
#import pandas as pd
import numpy as np
#from mlens.ensemble import SuperLearner
#from sklearn.linear_model import LogisticRegression
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.models import Sequential
#from keras.layers import Dense


accuracyTest = []
accuracyTrain = []

j = 200


knn = KNeighborsClassifier(n_neighbors=3)

for i in range(j): 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle = True)
    

## K Neighbors Classifier

    knn.fit(X_train, y_train)
    y_pred_Test = knn.predict(X_test)
    y_pred_Train = knn.predict(X_train)

    AccuracyTest = metrics.accuracy_score(y_test, y_pred_Test)
    AccuracyTrain = metrics.accuracy_score(y_train, y_pred_Train)

    
    accuracyTest.append(AccuracyTest)
    accuracyTrain.append(AccuracyTrain)
    
accuracyTrain_final = np.mean(accuracyTrain)
accuracyTest_final = np.mean(accuracyTest)

print('accuracy score of train dataset Final (KNN):', accuracyTrain_final)
print('accuracy score of test dataset Final (KNN):', accuracyTest_final)


# KNN with kFold = 7, cross Validation

print('Cross validation score for KNN:', cross_val_score(knn, X, y, cv=7, scoring='accuracy').mean())
print('Cross validation score for KNN:', cross_val_score(knn, X, y, cv=7, scoring='accuracy').std())

### Radius Neighbors Classifier

RadiusNeigh = RadiusNeighborsClassifier(algorithm='auto', leaf_size=30, metric='chebyshev',
                          metric_params=None, n_jobs=None, outlier_label=None,
                          p=1, radius=3.0, weights='distance')

for i in range(j): 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle = True)
    

## K Neighbors Classifier

    RadiusNeigh.fit(X_train, y_train)
    y_pred_Test = RadiusNeigh.predict(X_test)
    y_pred_Train = RadiusNeigh.predict(X_train)

    AccuracyTest = metrics.accuracy_score(y_test, y_pred_Test)
    AccuracyTrain = metrics.accuracy_score(y_train, y_pred_Train)

    
    accuracyTest.append(AccuracyTest)
    accuracyTrain.append(AccuracyTrain)
    
accuracyTrain_final = np.mean(accuracyTrain)
accuracyTest_final = np.mean(accuracyTest)

print('accuracy score of train dataset Final (Radius Neighbors):', accuracyTrain_final)
print('accuracy score of test dataset Final (Radius Neighbors):', accuracyTest_final)


# KNN with kFold = 7, cross Validation

print('Cross validation score for Radius Neighbors:', cross_val_score(RadiusNeigh,
                                                                      X, y, cv=7, scoring='accuracy').mean())


## SVC hyperparameter tuning
#from sklearn.model_selection import GridSearchCV 
#param_grid = {'C': [0.1, 1, 10, 100, 1000],  
#              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
#              'kernel': ['rbf','linear','poly']}  
#  
#grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) 
#  
## fitting the model for grid search 
#grid.fit(X ,y) 
#
## print best parameter after tuning 
#print(grid.best_params_) 
#  
## print how our model looks after hyper-parameter tuning 
#print(grid.best_estimator_) 
#kernel='rbf', C=100, gamma = 0.01

CLFsvc = svm.SVC(kernel='rbf', C=100, gamma = 0.01) #kernel = 'poly', degree = 3 , C = 0.9, gamma = 'scale', decision_function_shape= 'ovr')
#decision_function_shape='ovo', 'ovr'
for i in range(j): 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle = True)
    

## K Neighbors Classifier

    CLFsvc.fit(X_train, y_train)
    y_pred_Test = CLFsvc.predict(X_test)
    y_pred_Train = CLFsvc.predict(X_train)

    AccuracyTest = metrics.accuracy_score(y_test, y_pred_Test)
    AccuracyTrain = metrics.accuracy_score(y_train, y_pred_Train)

    
    accuracyTest.append(AccuracyTest)
    accuracyTrain.append(AccuracyTrain)
    
accuracyTrain_final = np.mean(accuracyTrain)
accuracyTest_final = np.mean(accuracyTest)

print('accuracy score of train dataset Final (SVC):', accuracyTrain_final)
print('accuracy score of test dataset Final (SVC):', accuracyTest_final)

# KNN with kFold = 7, cross Validation

print('Cross validation score for SVC:', cross_val_score(CLFsvc, X, y, cv=7, scoring='accuracy').mean())



# Random Forest Classifier

#Tunning hyperparameters
## Random Forest hyperparameter tuning
#from sklearn.model_selection import GridSearchCV 
#param_grid = { 
#    'n_estimators': [1, 200, 10],
#    'max_features': ['auto', 'sqrt', 'log2'],
#    'max_depth' : [5,150,1],
#    'min_samples_split' : [2, 5, 10],
#    'min_samples_leaf' : [1, 2, 4],
#    'bootstrap' : [True, False]
#}
#  
#grid = GridSearchCV(RandomForestClassifier(), param_grid, refit = True, verbose = 3) 
#  
## fitting the model for grid search 
#grid.fit(X ,y) 
#
## print best parameter after tuning 
#print(grid.best_params_) 
#  
## print how our model looks after hyper-parameter tuning 
#print(grid.best_estimator_) 





#CLFRFC = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
#                       criterion='entropy', max_depth=50, max_features='auto',
#                       max_leaf_nodes=None, max_samples=None,
#                       min_impurity_decrease=0.0, min_impurity_split=None,
#                       min_samples_leaf=1, min_samples_split=2,
#                       min_weight_fraction_leaf=0.0, n_estimators=50,
#                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
#                       warm_start=False)

#RFC using with the obtained hyperparameters
CLFRFC = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='entropy', max_depth=20, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=50,
                       n_jobs=None, oob_score=False, random_state=0,
                       verbose=0, warm_start=False)

for i in range(j): 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle = True)
    

## Random Forest Classifier training and fitting

    CLFRFC.fit(X_train, y_train)
    y_pred_Test = CLFRFC.predict(X_test)
    y_pred_Train = CLFRFC.predict(X_train)

    AccuracyTest = metrics.accuracy_score(y_test, y_pred_Test)
    AccuracyTrain = metrics.accuracy_score(y_train, y_pred_Train)

    
    accuracyTest.append(AccuracyTest)
    accuracyTrain.append(AccuracyTrain)
    
accuracyTrain_final = np.mean(accuracyTrain)
accuracyTest_final = np.mean(accuracyTest)

print('accuracy score of train dataset Final (RFC):', accuracyTrain_final)
print('accuracy score of test dataset Final (RFC):', accuracyTest_final)

# RF Classification with kFold = 7, cross Validation

print('Cross validation score for Random forest classifier:', cross_val_score(CLFRFC, X, y, cv=7, scoring='accuracy').mean())
scores = cross_val_score(CLFRFC, X, y, cv=7)

#####
## Logistics Regression Classifier
CLFLogReg = LogisticRegression(penalty='l2',solver = 'lbfgs', C = 0.09)
#CLFLogReg = LogisticRegression(multi_class='multinomial', random_state=1)
#####

for i in range(j): 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle = True)
    

## Logistics Classifier

    CLFLogReg.fit(X_train, y_train)
    y_pred_Test = CLFLogReg.predict(X_test)
    y_pred_Train = CLFLogReg.predict(X_train)

    AccuracyTest = metrics.accuracy_score(y_test, y_pred_Test)
    AccuracyTrain = metrics.accuracy_score(y_train, y_pred_Train)

    
    accuracyTest.append(AccuracyTest)
    accuracyTrain.append(AccuracyTrain)
    
accuracyTrain_final = np.mean(accuracyTrain)
accuracyTest_final = np.mean(accuracyTest)

print('accuracy score of train dataset Final (LogReg):', accuracyTrain_final)
print('accuracy score of test dataset Final (LogReg):', accuracyTest_final)

#Logistics Regression with kFold = 7, cross Validation

print('Cross validation score for LogReg classifier:', cross_val_score(CLFLogReg, X, y, cv=7, scoring='accuracy').mean())


## Guassian process Classifier
kernel = 10.0 * RBF(1)
GPClf = GaussianProcessClassifier(kernel=kernel)

for i in range(j): 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle = True)
    

    GPClf .fit(X_train, y_train)
    y_pred_Test = GPClf .predict(X_test)
    y_pred_Train = GPClf .predict(X_train)

    AccuracyTest = metrics.accuracy_score(y_test, y_pred_Test)
    AccuracyTrain = metrics.accuracy_score(y_train, y_pred_Train)

    
    accuracyTest.append(AccuracyTest)
    accuracyTrain.append(AccuracyTrain)
    
accuracyTrain_final = np.mean(accuracyTrain)
accuracyTest_final = np.mean(accuracyTest)

print('accuracy score of train dataset Final (Gaussian Process Classifier):', accuracyTrain_final)
print('accuracy score of test dataset Final (Gaussian Process Classifier):', accuracyTest_final)

### Guassian process Classifier  with kFold = 7, cross Validation

print('Cross validation score for Gaussian Process Classifier:', 
      cross_val_score(GPClf, X, y, cv=7, scoring='accuracy').mean())



## Multi-layer Perceptron  Classifier

#MLPClf = MLPClassifier(random_state=0, max_iter=500)

MLPClf = MLPClassifier(activation='relu', alpha=0.001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(60,60,60), learning_rate='constant',
              learning_rate_init=0.003, max_fun=15000, max_iter=500,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=0, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)

for i in range(j): 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle = True)
    

    MLPClf .fit(X_train, y_train)
    y_pred_Test = MLPClf .predict(X_test)
    y_pred_Train = MLPClf .predict(X_train)

    AccuracyTest = metrics.accuracy_score(y_test, y_pred_Test)
    AccuracyTrain = metrics.accuracy_score(y_train, y_pred_Train)

    
    accuracyTest.append(AccuracyTest)
    accuracyTrain.append(AccuracyTrain)
    
accuracyTrain_final = np.mean(accuracyTrain)
accuracyTest_final = np.mean(accuracyTest)

print('accuracy score of train dataset Final (Multi-layer Perceptron Classifier):', accuracyTrain_final)
print('accuracy score of test dataset Final (Multi-layer Perceptron Classifier):', accuracyTest_final)

###  Multi-layer Perceptron Classifier  with kFold = 7, cross Validation

print('Cross validation score for Multi-layer Perceptron Classifier:', 
      cross_val_score(MLPClf, X, y, cv=7, scoring='accuracy').mean())

print('Cross validation score for Multi-layer Perceptron Classifier:', 
      cross_val_score(MLPClf, X, y, cv=7, scoring='accuracy').std())
####
#AdaBoostClassifier
####

AdaBoostClf = AdaBoostClassifier(n_estimators=500, random_state=None,learning_rate=0.01,algorithm='SAMME.R')

for i in range(j): 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle = True)
    

    AdaBoostClf .fit(X_train, y_train)
    y_pred_Test =AdaBoostClf .predict(X_test)
    y_pred_Train = AdaBoostClf .predict(X_train)

    AccuracyTest = metrics.accuracy_score(y_test, y_pred_Test)
    AccuracyTrain = metrics.accuracy_score(y_train, y_pred_Train)

    
    accuracyTest.append(AccuracyTest)
    accuracyTrain.append(AccuracyTrain)
    
accuracyTrain_final = np.mean(accuracyTrain)
accuracyTest_final = np.mean(accuracyTest)

print('accuracy score of train dataset Final (Ada Boost Classifier):', accuracyTrain_final)
print('accuracy score of test dataset Final (Ada Boost Classifier):', accuracyTest_final)

###  Multi-layer Perceptron Classifier  with kFold = 7, cross Validation

print('Cross validation score for Ada Boost Classifier:', 
      cross_val_score(AdaBoostClf, X, y, cv=7, scoring='accuracy').mean())


#######
#Gaussian Naive Bayes (GaussianNB)
######

GNBClf = GaussianNB(var_smoothing=3)

for i in range(j): 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle = True)
    

    GNBClf .fit(X_train, y_train)
    y_pred_Test =GNBClf .predict(X_test)
    y_pred_Train = GNBClf .predict(X_train)

    AccuracyTest = metrics.accuracy_score(y_test, y_pred_Test)
    AccuracyTrain = metrics.accuracy_score(y_train, y_pred_Train)

    
    accuracyTest.append(AccuracyTest)
    accuracyTrain.append(AccuracyTrain)
    
accuracyTrain_final = np.mean(accuracyTrain)
accuracyTest_final = np.mean(accuracyTest)

print('accuracy score of train dataset Final (GaussianNB Classifier):', accuracyTrain_final)
print('accuracy score of test dataset Final (GaussianNB Classifier):', accuracyTest_final)

###  Gaussian Naive Bayes Classifier  with kFold = 7, cross Validation

print('Cross validation score for GaussianNB Classifier:', 
      cross_val_score(GNBClf, X, y, cv=7, scoring='accuracy').mean())

scoresGaussianNB = cross_val_score(GNBClf, X, y, cv=7)


#######
#Linear Discriminant Analysis
######

LDAClf = LinearDiscriminantAnalysis(n_components=2, priors=None, shrinkage='auto',
                           solver='lsqr', store_covariance=True, tol=0.0001)

for i in range(j): 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle = True)
    

    LDAClf .fit(X_train, y_train)
    y_pred_Test =LDAClf .predict(X_test)
    y_pred_Train = LDAClf .predict(X_train)

    AccuracyTest = metrics.accuracy_score(y_test, y_pred_Test)
    AccuracyTrain = metrics.accuracy_score(y_train, y_pred_Train)

    
    accuracyTest.append(AccuracyTest)
    accuracyTrain.append(AccuracyTrain)
    
accuracyTrain_final = np.mean(accuracyTrain)
accuracyTest_final = np.mean(accuracyTest)

print('accuracy score of train dataset Final (LDA Classifier):', accuracyTrain_final)
print('accuracy score of test dataset Final (LDA Classifier):', accuracyTest_final)

### Linear Discriminant Analysis Classifier  with kFold = 7, cross Validation

print('Cross validation score for LDA Classifier:', 
      cross_val_score(LDAClf, X, y, cv=7, scoring='accuracy').mean())

scoresLDA = cross_val_score(LDAClf, X, y, cv=7)



######
##Bagging Classifier
######

from sklearn.ensemble import BaggingClassifier
baggingCLF = BaggingClassifier(MLPClf,
                            max_samples=0.5, max_features=0.5)
#KNeighborsClassifier acc = 57.1
for i in range(j): 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle = True)
    

    baggingCLF .fit(X_train, y_train)
    y_pred_Test =baggingCLF .predict(X_test)
    y_pred_Train = baggingCLF .predict(X_train)

    AccuracyTest = metrics.accuracy_score(y_test, y_pred_Test)
    AccuracyTrain = metrics.accuracy_score(y_train, y_pred_Train)

    
    accuracyTest.append(AccuracyTest)
    accuracyTrain.append(AccuracyTrain)
    
accuracyTrain_final = np.mean(accuracyTrain)
accuracyTest_final = np.mean(accuracyTest)

print('accuracy score of train dataset Final (Bagging Classifier):', accuracyTrain_final)
print('accuracy score of test dataset Final (Bagging Classifier):', accuracyTest_final)

###  Bagging Classifier  with kFold = 7, cross Validation

print('Cross validation score for Bagging Classifier:', 
      cross_val_score(baggingCLF, X, y, cv=7, scoring='accuracy').mean())

scoresBaggingClf = cross_val_score(baggingCLF, X, y, cv=7)
y_predCross = cross_val_predict(baggingCLF, X, y, cv=7)



########
## Extra Tree Classification
########

ExtraTreesCLF = ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                     criterion='gini', max_depth=None, max_features='auto',
                     max_leaf_nodes=None, max_samples=None,
                     min_impurity_decrease=0.0, min_impurity_split=None,
                     min_samples_leaf=1, min_samples_split=2,
                     min_weight_fraction_leaf=0.0, n_estimators=50,
                     n_jobs=None, oob_score=False, random_state=0, verbose=0,
                     warm_start=False)

for i in range(j): 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle = True)
    

    ExtraTreesCLF .fit(X_train, y_train)
    y_pred_Test =ExtraTreesCLF .predict(X_test)
    y_pred_Train = ExtraTreesCLF .predict(X_train)

    AccuracyTest = metrics.accuracy_score(y_test, y_pred_Test)
    AccuracyTrain = metrics.accuracy_score(y_train, y_pred_Train)

    
    accuracyTest.append(AccuracyTest)
    accuracyTrain.append(AccuracyTrain)
    
accuracyTrain_final = np.mean(accuracyTrain)
accuracyTest_final = np.mean(accuracyTest)

print('accuracy score of train dataset Final (Extra Trees Classifier):', accuracyTrain_final)
print('accuracy score of test dataset Final (Extra Trees Classifier):', accuracyTest_final)

###  Extra Trees Classifier  with kFold = 7, cross Validation

print('Cross validation score for Extra Trees Classifier:', 
      cross_val_score(ExtraTreesCLF, X, y, cv=7, scoring='accuracy').mean())

scoresExtraTreesCLF = cross_val_score(ExtraTreesCLF, X, y, cv=7)


########
## Ensemble methods
########
from sklearn.ensemble import VotingClassifier

EnCLF = VotingClassifier(estimators=[
       ('knn', knn), ('SVC', CLFsvc), ('RF', CLFRFC), ('MLP',MLPClf)], voting='hard')

for i in range(j): 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle = True)
    

    EnCLF .fit(X_train, y_train)
    y_pred_Test =EnCLF .predict(X_test)
    y_pred_Train = EnCLF .predict(X_train)

    AccuracyTest = metrics.accuracy_score(y_test, y_pred_Test)
    AccuracyTrain = metrics.accuracy_score(y_train, y_pred_Train)

    
    accuracyTest.append(AccuracyTest)
    accuracyTrain.append(AccuracyTrain)
    
accuracyTrain_final = np.mean(accuracyTrain)
accuracyTest_final = np.mean(accuracyTest)

print('accuracy score of train dataset Final (Ensemble methods Classifier):', accuracyTrain_final)
print('accuracy score of test dataset Final (Ensemble methods Classifier):', accuracyTest_final)

###  Ensemble methods Classifier  with kFold = 7, cross Validation

print('Cross validation score for Ensemble methods Classifier:', 
      cross_val_score(EnCLF, X, y, cv=7, scoring='accuracy').mean())

scoresEnCLF = cross_val_score(EnCLF, X, y, cv=7)
y_predictCross = cross_val_predict(EnCLF, X, y , cv=7)

y_PredictCross7Fold = []
for k in range(2,8):
    result = cross_val_score(EnCLF, X, y, cv=k, scoring='accuracy')
    print(k, result.mean())
    y_pred = cross_val_predict(EnCLF, X, y, cv=k)
    print(y_pred)
    
y_PredictCross7Fold.append(y_pred)
y_PredictCross7Fold = np.array(y_PredictCross7Fold)
y_PredictCross7Fold = y_PredictCross7Fold.T