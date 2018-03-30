# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 12:49:30 2018

@author: Homagni
"""
'''
REFERECES USED:
    http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_hastie_10_2.html#sphx-glr-auto-examples-ensemble-plot-adaboost-hastie-10-2-py
    http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
    http://scikit-learn.org/stable/modules/ensemble.html
    
'''
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

class ensembleLearners(object):
    def __init__(self,train,test,techniques,parameters):
        self.train=np.loadtxt(train,delimiter=',',skiprows=1) #First row is the metadata
        self.test=np.loadtxt(test,delimiter=',',skiprows=1) #First row is the metadata
        
        self.techniques=techniques
        self.parameters=parameters
        self.n_estimators = 400
        # A learning rate of 1. may not be optimal for both SAMME and SAMME.R
        self.learning_rate = 1.
    def processData(self):
        self.x_train=self.train[:,0:4]
        self.y_train=self.train[:,-1]
        
        self.x_test=self.test[:,0:4]
        self.y_test=self.test[:,-1]
        #self.a = self.a[~np.isnan(self.a)] #sample usage to remove nans
    def Dstump(self):
        self.dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
        self.dt_stump.fit(self.x_train, self.y_train)
        self.dt_stump_err = 1.0 - self.dt_stump.score(self.x_test, self.y_test)
        
    def makemodelandfit1(self): #make models and fit for task 1
        self.ada_discrete = AdaBoostClassifier(base_estimator=self.dt_stump,learning_rate=self.learning_rate,n_estimators=self.n_estimators,algorithm="SAMME") #Unfortunately scipy does not have samme.M1
        self.ada_discrete.fit(self.x_train, self.y_train)
        self.rf=RandomForestClassifier(n_estimators=self.n_estimators, max_depth=1, min_samples_split=2, min_samples_leaf=1) #Random forest based on decision stumps
        self.rf.fit(self.x_train,self.y_train)
        names=['ada_discrete','random_forest']
    def makemodelandfit2(self): #make models and fit for task 2
        self.NN=MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False)
        self.NN.fit(self.x_train,self.y_train)
        self.knn=KNeighborsClassifier(3)
        self.knn.fit(self.x_train,self.y_train)
        self.LR=LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
        self.LR.fit(self.x_train,self.y_train)
        self.NB=BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
        self.NB.fit(self.x_train,self.y_train)
        self.DT=DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
        self.DT.fit(self.x_train,self.y_train)
        
    def showresults(self,task):
        if(task==1):
            print("Testing random forest performance")
            rfscorestrain = cross_val_score(self.rf, self.x_train, self.y_train)
            print("The mean training accuracy on random forest ",rfscorestrain.mean())
            rfscorestest = cross_val_score(self.rf, self.x_test, self.y_test)
            print("The mean testing accuracy on random forest ",rfscorestest.mean())
            
            print("Testing adaboosted-decision stumps performance")
            adscorestrain = cross_val_score(self.ada_discrete, self.x_train, self.y_train)
            print("The mean testing accuracy on adaboosted-decision stumps ",adscorestrain.mean())
            adscorestest = cross_val_score(self.ada_discrete, self.x_test, self.y_test)
            print("The mean testing accuracy on adaboosted-decision stumps ",adscorestest.mean())
        elif(task==2):
            print("Testing neuralnetwork performance")
            rfscorestrain = cross_val_score(self.rf, self.x_train, self.y_train)
            print("The mean training accuracy on random forest ",rfscorestrain.mean())
            rfscorestest = cross_val_score(self.rf, self.x_test, self.y_test)
            print("The mean testing accuracy on random forest ",rfscorestest.mean())
            
            print("Testing adaboosted-decision stumps performance")
            adscorestrain = cross_val_score(self.ada_discrete, self.x_train, self.y_train)
            print("The mean testing accuracy on adaboosted-decision stumps ",adscorestrain.mean())
            adscorestest = cross_val_score(self.ada_discrete, self.x_test, self.y_test)
            print("The mean testing accuracy on adaboosted-decision stumps ",adscorestest.mean())
        
    def visualize(self):
        #%% adaboost
        fig = plt.figure()
        plt.title('Adaboosting and number of estimators')
        ax = fig.add_subplot(111)
        
        ada_discrete_err = np.zeros((self.n_estimators,))
        for i, y_pred in enumerate(self.ada_discrete.staged_predict(self.x_test)):
            ada_discrete_err[i] = zero_one_loss(y_pred, self.y_test)
        
        ada_discrete_err_train = np.zeros((self.n_estimators,))
        for i, y_pred in enumerate(self.ada_discrete.staged_predict(self.x_train)):
            ada_discrete_err_train[i] = zero_one_loss(y_pred, self.y_train)
        ax.plot(np.arange(self.n_estimators) + 1, ada_discrete_err,label='Discrete AdaBoost Test Error',color='red')
        ax.plot(np.arange(self.n_estimators) + 1, ada_discrete_err_train,label='Discrete AdaBoost Train Error',color='blue')
        
        ax.set_ylim((0.0, 0.5))
        ax.set_xlabel('n_estimators')
        ax.set_ylabel('error rate')
        leg = ax.legend(loc='upper right', fancybox=True)
        leg.get_frame().set_alpha(0.7)
        plt.show()
        #%%
        
        
        
        
el=ensembleLearners('lab3-train.csv','lab3-test.csv',['adaboost-DT'],['lr -1','n_est -400'])
el.processData()
el.Dstump()
el.makemodelandfit1()
el.showresults(1)
el.visualize()



