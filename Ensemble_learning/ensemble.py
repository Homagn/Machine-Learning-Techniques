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
import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier

np.random.seed(11)
class ensembleLearners(object):
    def __init__(self,train,test,techniques,parameters):
        self.train=np.loadtxt(train,delimiter=',',skiprows=1) #First row is the metadata
        self.test=np.loadtxt(test,delimiter=',',skiprows=1) #First row is the metadata
        
        self.techniques=techniques
        self.parameters=parameters
        self.n_estimators = 400 #used in RF type models
        self.n_neighbors=200 #used in knn type models
        self.names=[]
        self.models=[]
        # A learning rate of 1. may not be optimal for both SAMME and SAMME.R
        self.learning_rate = 0.1
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
        
    def makemodel1(self): #make models for task 1
        self.ada_discrete = AdaBoostClassifier(base_estimator=self.dt_stump,learning_rate=self.learning_rate,n_estimators=self.n_estimators,algorithm="SAMME") #Unfortunately scipy does not have samme.M1
        self.rf=RandomForestClassifier(n_estimators=self.n_estimators, max_depth=1, min_samples_split=2, min_samples_leaf=1) #Random forest based on decision stumps
        
        self.names.extend(['ada_discrete','random_forest'])
        self.models.extend([self.ada_discrete,self.rf])
    def makemodel2(self): #make models for task 2
        self.NN=MLPClassifier(hidden_layer_sizes=(150,100,50,), activation='logistic', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.01, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True)
        self.knn=KNeighborsClassifier(n_neighbors=self.n_neighbors, weights='distance',leaf_size=10, p=2, metric='minkowski')
        self.LR=LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.1, fit_intercept=True, intercept_scaling=1, class_weight='balanced', random_state=None, solver='liblinear', max_iter=200, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
        self.NB=BernoulliNB(alpha=0.8, binarize=0.02, fit_prior=False, class_prior=None)
        self.DT=DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=4, min_samples_split=100, min_samples_leaf=50, min_weight_fraction_leaf=0.0, max_features=4, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
        
        self.names.extend(['Neural networks','k-Nearest Neighbor','Logistic regression','Naive Bayes','Decision trees'])
        self.models.extend([self.NN,self.knn,self.LR,self.NB,self.DT])
    def makevoting(self,weights):
        a=copy.copy(self.names)
        b=copy.copy(self.models)
        if(not weights):
            self.vc=VotingClassifier([[a,b] for  a,b in zip(a, b)], voting='hard', weights=None, n_jobs=1, flatten_transform=None)
        else:
            self.vc=VotingClassifier([[a,b] for  a,b in zip(a, b)], voting='hard', weights=weights, n_jobs=1, flatten_transform=None)
        self.names.append('voting_classifier')
        self.models.append(self.vc)
    def fitmodels(self,start):
        if(start<len(self.models)):
            for i in range(start,len(self.models)):
                self.models[i].fit(self.x_train,self.y_train)
        elif(start==-1):
            self.models[-1].fit(self.x_train,self.y_train)
        else:
            print("provided wrong start index")
    def showresults(self,start):
        if(start<len(self.models)):
            for i in range (start,len(self.names)):
                print("testing "+self.names[i]+" performance")
                scoretrain=cross_val_score(self.models[i], self.x_train, self.y_train)
                print("The mean training accuracy on "+self.names[i]+" ",scoretrain.mean())
                scoretest = cross_val_score(self.models[i], self.x_test, self.y_test)
                print("The mean testing accuracy on "+self.names[i]+" ",scoretest.mean())
                print("Printing the confusion matrix for "+self.names[i])
                y_pred=self.models[i].predict(self.x_test)
                cf=confusion_matrix(self.y_test,y_pred)
                print(cf)
        elif(start==-1):
            print("testing "+self.names[i]+" performance")
            scoretrain=cross_val_score(self.models[-1], self.x_train, self.y_train)
            print("The mean training accuracy on "+self.names[-1]+" ",scoretrain.mean())
            scoretest = cross_val_score(self.models[-1], self.x_test, self.y_test)
            print("The mean testing accuracy on "+self.names[-1]+" ",scoretest.mean())
            print("Printing the confusion matrix for "+self.names[-1])
            y_pred=self.models[-1].predict(self.x_test)
            cf=confusion_matrix(self.y_test,y_pred)
            print(cf)
        else:
            print("provided wrong start index")
    def visualize(self):
        print("Getting a PCA visualization of the training data")
        pca = PCA(n_components=None)
        pca.fit(self.x_train)
        print("pca explained variance ratio")
        print(pca.explained_variance_ratio_)
        
        print("Getting a PCA visualization of the testing data")
        pca = PCA(n_components=None)
        pca.fit(self.x_test)
        print("pca explained variance ratio")
        print(pca.explained_variance_ratio_)
        
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
        
        print("printing relative importance of each feature using random forests")
        print(self.rf.feature_importances_)
        
        ax.set_ylim((0.0, 0.5))
        ax.set_xlabel('n_estimators')
        ax.set_ylabel('error rate')
        leg = ax.legend(loc='upper right', fancybox=True)
        leg.get_frame().set_alpha(0.7)
        plt.show()
        
        
        
        
        
el=ensembleLearners('lab3-train.csv','lab3-test.csv',['adaboost-DT'],['lr -1','n_est -400'])
el.processData()
el.Dstump()
el.makemodel1()
el.makemodel2()
el.makevoting([0.73,0.79,0.79,0.70,0.72,0.79,0.79]) #pass [] for uniform weighting and for ex [1,1,2] for the weights of 3 classifiers
el.fitmodels(0)
el.showresults(0)
el.visualize()

