import numpy as np
import math
import csv
import sys


class NB(object):

    def __init__(self, train_data, train_label, test_data, test_label, sortflag, test):

        self.docid=[]
        self.wordid=[]
        self.count=[]
        self.labels=[]
        self.sortflag=sortflag #flag to denote whether given dataset is sorted based on docid
        '''
        Parse and store the train label and the train data
        '''
        print("Parsing the csv files ")
        if(test==False):
            with open(train_data) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                for row in readCSV:
                    self.docid.append(int(row[0]))
                    self.wordid.append(int(row[1]))
                    self.count.append(int(row[2]))
            with open(train_label) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                for row in readCSV:
                    self.labels.append(int(row[0]))
        if(test==True):        
            with open(test_data) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                for row in readCSV:
                    self.docid.append(int(row[0]))
                    self.wordid.append(int(row[1]))
                    self.count.append(int(row[2]))
            with open(test_label) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                for row in readCSV:
                    self.labels.append(int(row[0]))
            print("Entered test mode, data is not sorted, so sorting the test data")
            self.sd=np.argsort(self.docid)
            self.ndocid=self.docid
            self.nwordid=self.wordid
            self.ncount=self.count
            for i in range(len(self.docid)):
                self.ndocid[i]=self.docid[self.sd[i]]
                self.nwordid[i]=self.wordid[self.sd[i]]
                self.ncount[i]=self.count[self.sd[i]]
            self.docid=self.ndocid
            self.wordid=self.nwordid
            self.count=self.ncount
                
        print("creating placeholders ")           
        self.sortindex=np.argsort(self.docid) #stores the indices of the sorted array in ascending order
        self.sortlabels=np.zeros(len(self.sortindex)) #appends the labels side by side the docid, wordid and count
        for i in range(len(self.docid)):
            self.sortlabels[self.sortindex[i]]=self.labels[self.docid[self.sortindex[i]]-1]
        
        self.a=np.amax(self.docid)
        self.b=np.amax(self.wordid)
        self.c=np.amax(self.labels)
        
        self.n=np.zeros((self.c))
        self.nk=np.zeros((self.c,self.b))
        self.ndk=np.zeros((self.b,self.a)) #calculates the number of occurances of word wk in document d
        self.nc=np.zeros(self.c)
        
        self.PMLE=np.ones((self.c,self.b))
        self.PBE=np.ones((self.c,self.b))
        
        k=0
        cat=1
        if(self.sortflag==True):
            print("Informed that data is sorted, hence finding index points ")
            self.docsortlist=np.zeros((self.a)+1)
#            print(np.shape(self.docsortlist))
            while k<len(self.docid):
                if(self.docid[k]>cat):
                    self.docsortlist[cat+1]=k
                    cat=cat+1
                k=k+1
            
    
    def _calcprob(self):
        
        
        
#        print("minimum value ",np.min(self.sortlabels))
#        print("Shape ",np.shape(self.sortlabels))
        
        for i in range(len(self.labels)):
            self.nc[int(self.labels[i])-1]=self.nc[int(self.labels[i])-1]+1
        '''
        Calulating the probability table
        '''
        for i in range(len(self.docid)):
            self.n[int(self.sortlabels[i])-1]=self.n[int(self.sortlabels[i])-1]+self.count[i] #total number of words in all documents in class omega j
            self.nk[int(self.sortlabels[i])-1,int(self.wordid[i])-1]=self.nk[int(self.sortlabels[i])-1,int(self.wordid[i])-1]+self.count[i] #number of times word wk occurs in all documents in class omega j
        
#        print("checking whether n has a 0 or not")
#        print(self.n)
        
        for i in range(len(self.wordid)):
            self.ndk[self.wordid[i]-1,self.docid[i]-1]=self.ndk[self.wordid[i]-1,self.docid[i]-1]+self.count[i]
            
        for i in range(self.c):
            self.PMLE[i,:]=self.nk[i,:]/self.n[i]
            self.PBE[i,:]=(self.nk[i,:]+np.ones((self.b)))/(self.n[i]+self.b)
        self.Pomega=(self.n)/len(self.sortlabels)
#        print("checking 0s in PBE ")
#        print(np.min(self.PBE))
    def _printEstimators(self):
#        print("Shape of PMLE ")
#        print(np.shape(self.PMLE))
#        print("Shape of PBE ")
#        print(np.shape(self.PBE))
#        print("printing PMLE ")
#        print(self.PMLE)
#        print("printing PBE ")
#        print(self.PBE)
        print("Printing the class priors ")
        for i in range(len(self.Pomega)):
            print("P(omega =%i ) ="%(i+1))
            print(self.Pomega[i])
            
    def _predict(self,startidx,numdocs,classifier_type,show_progress): #Bayesian methods are fast to train but take a lot of time to predict bacuse we have to find argmax of the class for which the likelihood is maximized
        '''
        We have to use the given formula (1) in the lab text.
        For each document the prediction will be the class that maximizes the value inside the argmax function
        with omega NB being the probability of that class happening in reality.
        '''
        self.startidx=startidx
        self.numdocs=numdocs
        if(classifier_type=='MLE'):
            self.PSELECT=self.PMLE
        if(classifier_type=='BE'):
            self.PSELECT=self.PBE
        if(numdocs==-1):
            numdocs=self.a
            startidx=0
        self.predClasses=np.zeros((self.a,self.c))
        
        for k in range(startidx,numdocs):
            if(show_progress==True):
                print("Document ID %i "%(k+1))
            for i in range(self.c): #create an array of proability of each class happening and chose the argmax as the prediction 
                pi=0
                if(self.sortflag==True):
                    for j in range(int(self.docsortlist[k]),int(self.docsortlist[k+1])):
                        try:
                            pi=pi+(self.count[j])*math.log(float(self.PSELECT[i,self.wordid[j]-1]),10)
                        except:
                            pi=pi
                            #print("Few data in test is not present in train ")
                else:    
                    for j in range(len(self.docid)):
                        if((k)==self.docid[j]-1):
                            try:
                                pi=pi+(self.count[j])*math.log(float(self.PSELECT[i,self.wordid[j]-1]),10)
                            except:
                                pi=pi
                                #print("Few data in test is not present in train ")
                pi=pi+math.log(self.Pomega[i],10)
                self.predClasses[k,i]=pi
        print("Printing the predictions for all the documents ")
        
                
    def _performance(self,indivStats):
        self.matchcount=0
        self.class_acc=np.zeros(self.c)
        self.confusion=np.zeros((self.c,self.c))
        for i in range(self.startidx,self.numdocs):
            if(indivStats==True):
                print("Document %i predicted class is %i actual class is %i "%((i+1),(np.argmax(self.predClasses[i,:])+1),self.labels[i]))
                print("Predicted Classes vector for the document is ")
                print(self.predClasses[i,:])
            if((np.argmax(self.predClasses[i,:])+1)==self.labels[i]):
                #print("match %i"%self.matchcount)
                self.matchcount=self.matchcount+1
                self.class_acc[self.labels[i]-1]=self.class_acc[self.labels[i]-1]+1
            self.confusion[(np.argmax(self.predClasses[i,:])),self.labels[i]-1]=self.confusion[(np.argmax(self.predClasses[i,:])),self.labels[i]-1]+1
        m=float(self.matchcount)
        t=float(self.numdocs-self.startidx)
        print("The total accuracy is ")
        print(100*(m/t))
        
        for i in range(self.c):
            print("Class %i accuracy is %f percent"%((i+1),100*float((self.class_acc[i])/self.nc[i])))
        print("Printing the confusion matrix")
        print(self.confusion)
                
            
    def _checkcsv(self):
        print(len(self.labels))
        print(self.docid[0:20])
        print("Now printing the labels ")
        print(self.labels[0:20])
        print(type(self.labels[0]))


def main():
    # print command line arguments
    try:
        td=sys.argv[3]
        tl=sys.argv[4]
        tdt=sys.argv[5]
        tlt=sys.argv[6]
    except:
        print("provide proper arguments along with the command")
        sys.exit(0)
    
    nbTrain=NB(td,tl,tdt,tlt, True, False)
    nbTest=NB(td,tl,tdt,tlt, True, True)
#    nbTrain._checkcsv()
    nbTrain._calcprob()
    nbTest._calcprob()
    nbTrain._printEstimators()
    print("Transferring the training knowledge on testing instance")
    nbTest.PMLE=nbTrain.PMLE
    nbTest.PBE=nbTrain.PBE
    nbTest.Pomega=nbTrain.Pomega
    
    print("started prediction process on training data for training done on training data")
    nbTrain._predict(1,11200,'BE',False)#Change this to true to monitor progress of prediction
    nbTrain._performance(False)
    print("The testing data has been found to contain a lot of outliers from the train set, also it is unsorted unlike the training data, so prediction will take some time. Thus only a part of testing is done and may have some issues with accuracy and confusion matrix")
    print("started prediction process on testing data for training done on training data")
    nbTest._predict(1,7505,'MLE',False)#Change this to true to monitor progress of prediction
    nbTest._performance(False)#Change this to true to view individual predictions
    

    
        

if __name__ == "__main__":
    main()

