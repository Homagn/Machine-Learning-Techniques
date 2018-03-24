# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 12:49:30 2018

@author: Homagni
"""
'''
REFERECES USED:
    http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_hastie_10_2.html#sphx-glr-auto-examples-ensemble-plot-adaboost-hastie-10-2-py
    http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
    
'''
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier


class ensembleLearners(object):
    def __init__(self,data,techniques,parameters):
        self.data=np.loadtxt(data,delimiter=',',skiprows=1) #First row is the metadata
        self.techniques=techniques
        self.parameters=parameters
        self.n_estimators = 400
        # A learning rate of 1. may not be optimal for both SAMME and SAMME.R
        self.learning_rate = 1.
    def processData(self):
        datalen=len(self.data)
        split=int(datalen/3) #Use first 1/3 rd data as test data
        self.x_train=self.data[split:,0:4]
        self.y_train=self.data[split:,-1]
        
        self.x_test=self.data[0:split,0:4]
        self.y_test=self.data[0:split,-1]
        #self.a = self.a[~np.isnan(self.a)] #sample usage to remove nans
    def Dstump(self):
        self.dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
        self.dt_stump.fit(self.x_train, self.y_train)
        self.dt_stump_err = 1.0 - self.dt_stump.score(self.x_test, self.y_test)
    def Ada(self):
        self.ada_discrete = AdaBoostClassifier(base_estimator=self.dt_stump,learning_rate=self.learning_rate,n_estimators=self.n_estimators,algorithm="SAMME") #Unfortunately scipy does not have samme.M1
        self.ada_discrete.fit(self.x_train, self.y_train)
    def visualize(self):
        fig = plt.figure()
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

el=ensembleLearners('transfusion.txt',['adaboost-DT'],['lr -1','n_est -400'])
el.processData()
el.Dstump()
el.Ada()
el.visualize()


