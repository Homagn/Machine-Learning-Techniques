# -*- coding: utf-8 -*-
"""
Created on Sun Mar 04 15:29:42 2018

@author: Homagni

BASIC CNN class 
"""

from matplotlib import pyplot
from scipy.misc import toimage

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD,RMSprop
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.utils import plot_model
import matplotlib.pyplot as plt
K.set_image_dim_ordering('th')

seed = 7
np.random.seed(seed)

class NeuralNets(object):
    def __init__(self,txtfl1,txtfl2,params):
        print("Initialized")
        self.TrainDat=np.loadtxt(txtfl1,delimiter=',')
        self.TestDat=np.loadtxt(txtfl2,delimiter=',')
        self.params=params
    
    def _loadtxtfile(self,testload):
        self.imagedim=int(np.sqrt(len(self.TrainDat[1]-1)))
        self.x_train=np.zeros((len(self.TrainDat),1,self.imagedim,self.imagedim))
        self.x_trainf=np.zeros((len(self.TrainDat),len(self.TrainDat[1])-1))
        self.y_train=np.zeros((len(self.TrainDat),))
        #Parsing train text file
        for i in range (len(self.TrainDat)):
            self.x_trainf[i]= self.TrainDat[i,0:-1] #Need a vector when training fully connected nn
            self.x_train[i] = np.reshape(self.TrainDat[i,0:-1],(1,self.imagedim,self.imagedim)) #Channel,image_dim,image_dim
            self.y_train[i] = self.TrainDat[i,-1]
        #Parsing test text file    
        self.x_test=np.zeros((len(self.TestDat),1,self.imagedim,self.imagedim))
        self.x_testf=np.zeros((len(self.TestDat),len(self.TestDat[1])-1))
        self.y_test=np.zeros((len(self.TestDat),))
        for i in range (len(self.TestDat)):
            self.x_test[i] = np.reshape(self.TestDat[i,0:-1],(1,self.imagedim,self.imagedim))
            self.x_testf[i] = self.TestDat[i,0:-1] #Need a vector when training fully connected nn
            self.y_test[i] = self.TestDat[i,-1]
        self.nb_classes = len(np.unique(self.y_train))
        #Convert the labels to one hot encoding
        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_test = np_utils.to_categorical(self.y_test)
        if testload==True:
            print("testing load",self.x_train.shape)
            print("testing load",self.y_train.shape)
            print("testing load",self.x_train[1])
            print("testing load",self.y_train[1])
            
        if(self.params[1]=='input scaled'):
            self.x_train=self.x_train/(np.max(self.x_train))
            self.x_trainf=self.x_trainf/(np.max(self.x_train))  
            
    def _visualize(self):
        # create a grid of 3x3 images
        for i in range(0, 9):
        	pyplot.subplot(330 + 1 + i)
        	pyplot.imshow(toimage(self.x_train[i,0,:]))
        # show the plot
        pyplot.show()
        
    def _CNNlayer(self):
        self.model1=Sequential()
        self.model1.add(Conv2D(self.imagedim, (3, 3), input_shape=(1, self.imagedim, self.imagedim), padding='same', activation='relu', kernel_constraint=maxnorm(3)))#This is not a hidden layer
        self.model1.add(Dropout(0.2))
        self.model1.add(Conv2D(self.imagedim, (3, 3), activation=self.params[3], padding='same', kernel_constraint=maxnorm(3)))
        self.model1.add(MaxPooling2D(pool_size=(2, 2)))
        self.model1.add(Flatten())
        return self.model1
    
    def _FCNlayer(self):
        # Create the model
        self.model2 = Sequential()
        dfirst=self.imagedim**2
        if(self.params[0]=='usecnn'):
            self.model2.add(self._CNNlayer())
            dfirst=512
        else:
            self.model2.add(Dense(dfirst, input_shape=(dfirst,), kernel_initializer='normal', activation='relu'))# This is not a hidden layer
        '''
        Beginning of hidden layer
        '''
        for i in range(int(self._valueafterdash(self.params[4]))):
            self.model2.add(Dense(int(self._valueafterdash(self.params[5])), activation=self.params[3]))
        self.model2.add(Dropout(0.5))
        self.model2.add(Dense(self.nb_classes, activation=self.params[10]))
        return self.model2
    
    def _createModel(self):    
        # Compile model
        self.model = Sequential()
        self.model.add(self._FCNlayer())
        self.epochs = int(self._valueafterdash(self.params[12]))
        self.lrate = self._valueafterdash(self.params[6])
        self.decay = self.lrate/self.epochs
        self.sgd = SGD(lr=self.lrate, momentum=self._valueafterdash(self.params[7]), decay=self.decay, nesterov=False)
        if(self.params[11]=='sgd'):
            self.model.compile(loss=self.params[2], optimizer=self.sgd, metrics=['accuracy'])
        else:
            self.rmspr=RMSprop(lr=self.lrate, rho=self._valueafterdash(self.params[7]), epsilon=None, decay=0.0)
            self.model.compile(loss=self.params[2], optimizer=self.rmspr, metrics=['accuracy'])
        print(self.model.summary())
        print("Model created Successfully ")
        
    def _valueafterdash(self,a):
        pos=a.find('-')
        value=float(a[-(len(a)-pos-1):])
        return value
    
    def _train(self):
        # Fit the model
        self.history=[]
        self.model.trainable=True
        self.model_name=" ".join(self.params)
        model_json = self.model.to_json()
        with open("model/"+self.model_name+".json", "w") as json_file:
            json_file.write(model_json)
        print("started to train ")
        if self.epochs>0:
            if(self.params[0]=='usecnn'):
                hist=self.model.fit(self.x_train, self.y_train, validation_split=0.33, batch_size=int(self._valueafterdash(self.params[13])),epochs=self.epochs,verbose=1)
                self.history.append(hist)
                scores = self.model.evaluate(self.x_test, self.y_test, verbose=0)
            else:
                hist=self.model.fit(self.x_trainf, self.y_train, validation_split=0.33, batch_size=int(self._valueafterdash(self.params[13])),epochs=self.epochs,verbose=1)
                self.history.append(hist)
                scores = self.model.evaluate(self.x_testf, self.y_test, verbose=0)
        self.model.save_weights("weights/"+self.model_name)
        # Final evaluation of the model
        
        print("Accuracy: %.2f%%" % (scores[1]*100))
    
    def _plot(self):
        fig = plt.figure()
        xx=self.history[0].history['acc']
        yy=self.history[0].history['val_acc']
#        for i in range(int(self._valueafterdash(self.params[12])-1)):
#            x=self.history[i+1].history['acc']
#            xx=np.concatenate((xx,x),axis=0)
#            y=self.history[i+1].history['val_acc']
#            yy=np.concatenate((yy,y),axis=0)
        plt.plot(xx)
        plt.plot(yy)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('Performance/'+self.model_name+'.png')   # save the figure to file
        plt.close(fig)

passparam=['usecnn','input scaled','categorical_crossentropy','relu','hidden layers -3','hidden units -8','learning rate -0.01','momentum -0.9','filter size-3','maxpooling size -2','softmax','sgd','epochs -25','batchsize -10']
passparam[0]='usecnn' #HEAD PRIMARY CATEGORY
passparam[1]='input not scaled'
passparam[2]='categorical_crossentropy' # PRIMARY CATEGORY
passparam[3]='relu' #PRIMARY CATEGORY
passparam[4]='hidden layers -3'
passparam[5]='hidden units -50'
passparam[6]='learning rate -0.001'
passparam[7]='momentum -0.9'
passparam[8]='square filter size-3'
passparam[9]='maxpooling size -2'
passparam[10]='sigmoid'
passparam[11]='sgd'
passparam[12]='epochs -50'
passparam[13]='batchsize -100'

nn=NeuralNets('optdigits.tra','optdigits.tes',passparam)
nn._loadtxtfile(True)
nn._visualize()
nn._createModel()
nn._train()
nn._plot()