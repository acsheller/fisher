'''
This module is intended to perform fisher multicast feature ranking.



'''

import numpy as np
import pandas as pd
class FisherMultiClassFeatureRanker(object):
    '''
    '''

    def fisherMultiClassFeatureRankingDigits(self,data=None,labels=None,method=1):
        '''
        Python version of Matlab fisherMultClassFeatureRanking
        '''
        alg = {}
        X = data
        Y = labels
        numEx = data.shape[0]
        vDim  = data.shape[1]
        oDim = max(Y)
        
        alg['rank'] = []
        
        # Get the indexes of numbers
        zeroLabels = np.nonzero(labels.values == 0)
        oneLabels  = np.nonzero(labels.values == 1)
        twoLabels = np.nonzero(labels.values == 2)
        threeLabels = np.nonzero(labels.values == 3)
        fourLabels = np.nonzero(labels.values == 4)
        fiveLabels = np.nonzero(labels.values == 5)
        sixLabels = np.nonzero(labels.values == 6)
        sevenLabels = np.nonzero(labels.values == 7)
        eightLabels = np.nonzero(labels.values == 8)
        nineLabels = np.nonzero(labels.values == 9)
    
        STDs = []
        MEANs = []    
        if oDim < 2: # if one-class # not doing this yet
            pass
        
        else:
            
            STDs =(np.std(np.array(X.values)[oneLabels].T,1))
            STDs = np.add(STDs,np.std(np.array(X.values)[twoLabels].T,1))
            STDs = np.add(STDs,np.std(np.array(X.values)[threeLabels].T,1))  
            STDs = np.add(STDs,np.std(np.array(X.values)[fourLabels].T,1))
            STDs = np.add(STDs,np.std(np.array(X.values)[fiveLabels].T,1)) 
            STDs = np.add(STDs,np.std(np.array(X.values)[sixLabels].T,1)) 
            STDs = np.add(STDs,np.std(np.array(X.values)[sevenLabels].T,1)) 
            STDs = np.add(STDs,np.std(np.array(X.values)[eightLabels].T,1)) 
            STDs = np.add(STDs,np.std(np.array(X.values)[nineLabels].T,1)) 
            #STDs = np.add(STDs,np.std(np.array(X.values)[zeroLabels].T,1))   
            STDs = np.array(STDs)
            MEANs.append(np.mean(np.array(X.values)[oneLabels].T,1).T)
            MEANs.append(np.mean(np.array(X.values)[twoLabels].T,1).T)
            MEANs.append(np.mean(np.array(X.values)[threeLabels].T,1).T)
            MEANs.append(np.mean(np.array(X.values)[fourLabels].T,1).T)
            MEANs.append(np.mean(np.array(X.values)[fiveLabels].T,1).T)
            MEANs.append(np.mean(np.array(X.values)[sixLabels].T,1).T)
            MEANs.append(np.mean(np.array(X.values)[sevenLabels].T,1).T)
            MEANs.append(np.mean(np.array(X.values)[eightLabels].T,1).T)
            MEANs.append(np.mean(np.array(X.values)[nineLabels].T,1).T)
            #MEANs.append(np.mean(np.array(X.values)[zeroLabels].T,1).T)
            MEANs = np.array(MEANs)
      
            MS = np.array(MEANs/STDs)
            indTemp  = []
            corrTemp = []
            coorSort = []
            print()
            for i in range(0,oDim):
                WW1 = np.delete(MS.T,[i],axis=1)
                
                WW1a = np.array([MS[i]]*(oDim-1) )
                
                WWFinal = np.subtract(WW1a.T,WW1) #<---
                
                if method == 1:
                    rankW = np.min(WWFinal,axis=1)
                else:
                    print("Not Yet Implemented for cases 2, 3, 4")
            
                u = np.sort(-np.abs(rankW.T))
                v = np.argsort(-np.abs(rankW.T))
                indTemp.append(v)
                corrTemp.append(u)
            indTemp = np.array(indTemp).T.reshape(1,oDim*vDim)
            corrTemp = np.array(corrTemp).T.reshape(1,oDim*vDim)
            corrSort = np.sort(corrTemp)
            indx = np.argsort(corrTemp)
            indxSort = indTemp.T[indx]
            indTemp = np.fliplr(indxSort)
            corrTemp = np.fliplr(corrSort)
            [uu,vv] = np.unique(indTemp,return_index=True)
            w = np.sort(vv)
            s = np.argsort(vv)
    
            values = corrTemp.T[w][::-1]
            rankIndex = uu[s][::-1]
            self.model= {}
            self.model['featureIndex'] = rankIndex
            self.model['rankValue'] = values
            self.model['featureRankingMethod'] = 'fishersMultiClassFeatureRanking'


    def fisherMultiClassFeatureRankingIris(self,data=None,labels=None,method=1):
        '''
        Python version of Matlab fisherMultClassFeatureRanking
        Currently does not work with just two classes.
        big-O = O(nlogn) sorting is expensive 
        
        Big-T = 10 + 8*n + 4 + 3*(4+2n+2*nlogn)+3n+10+2*nlogn
              = 36 +17n +8nlogn
        '''
        
        X = data                                                                      #1
        Y = labels                                                                    #1
        # Number of features
        vDim  = data.shape[1]                                                          #1
        
        # The number of unique labels in our data
        oDim = np.unique(Y).shape[0]                                                   #1
        
        #Seperate out the labels for each class
        setosaLabels = np.nonzero(Y.values == 'setosa')                                #1
        versicolorLabels  = np.nonzero(Y.values == 'versicolor')                       #1
        virginicaLabels = np.nonzero(Y.values == 'virginica')                          #1
        
        STDs = []                                                                      #1
        MEANs = []                                                                     #1
        if oDim < 2: # if two-class # not doing this yet
            print("Sorry not yet implemented")                                         #1
        
        # We have more than two classes
        else:
            # Get Standard Deviations for each species seperately
            # And then add back to the array
            STDs =(np.std(np.array(X.values)[setosaLabels].T,1))                       #n
            STDs = np.add(STDs,np.std(np.array(X.values)[versicolorLabels].T,1))       #n
            STDs = np.add(STDs,np.std(np.array(X.values)[virginicaLabels].T,1))        #n 
            STDs = np.array(STDs)                                                      #n
            # Get the means for each seperately                                              
            # Adding to MEANS array as its done
            MEANs.append(np.mean(np.array(X.values)[setosaLabels].T,1).T)             #n
            MEANs.append(np.mean(np.array(X.values)[versicolorLabels].T,1).T)         #n
            MEANs.append(np.mean(np.array(X.values)[virginicaLabels].T,1).T)          #n
            
            # Make sure its a numpy array
            MEANs = np.array(MEANs)                                                    #n 8
            
            #MS is Mean Over Standard Deviation
            MS = np.array(MEANs/STDs)                                                   #1
            
            # Some storage for next part of the process
            indTemp  = []                                                                #1
            corrTemp = []                                                                #1
            coorSort = []                                                               #1
            
            # oDim is the number of unique labels in our data
            for i in range(0,oDim):                                                  #3*(4+2n+2*nlogn)
                # Take away the i'th column
                WW1 = np.delete(MS.T,[i],axis=1)                                       #1
                # Take that one column that was just deleted and 
                # give it correct dimensions
                WW1a = np.array([MS[i]]*(oDim-1) )                                     #1
                # Subtract the results -- arrays need to be same shape
                WWFinal = np.subtract(WW1a.T,WW1)                                      #n   
                
                # Choose one of these methods, 1 is default
                if method == 1:                                                         #1 
                    rankW = np.min(WWFinal,axis=1)                                      #n
                elif method == 2:
                    rankW = np.max(WWFinal,axis=1)                                      
                elif method == 3:
                    rankW = np.sum(WWFinal,axis=1)
                elif method == 4:
                    min = np.min(WWFinal,axis=1)
                    max = np.max(WWFinal,axis=1)
                    rankW = (min+max/2)
                else:
                    sys.stderr.write("That wasn't an option \n")
                    break
                
                # Get the values sorted
                u = np.sort(-np.abs(rankW.T))                                            #n*logn
                # Get the index of the values before sorting
                v = np.argsort(-np.abs(rankW.T))                                         #n*logn
                
                # Assign both to a temp array
                indTemp.append(v)                                                        #1
                corrTemp.append(u)                                                       #1
            # Reshape things to specified dimension
            indTemp = np.array(indTemp).T.reshape(1,oDim*vDim)                           #n
            corrTemp = np.array(corrTemp).T.reshape(1,oDim*vDim)                         #n
            # Sort
            corrSort = np.sort(corrTemp)                                                 #n*logn
            # Get positions of things before they were sorted
            indx = np.argsort(corrTemp)                                                  #n*logn
            # Get the items at the index
            indxSort = indTemp.T[indx]                                                   #1
            # Flip it around
            indTemp = np.fliplr(indxSort)                                                #1
            # Flip it around
            corrTemp = np.fliplr(corrSort)                                               #1
            # np.unique returns the unique values of their index
            [uu,vv] = np.unique(indTemp,return_index=True)                               #n
            # Sort vv
            w = np.sort(vv)                                                              #n*logn
            # Get the index of soirting vv
            s = np.argsort(vv)                                                           #n*logn
            
            # sort the values in reverse order -- reverse them
            values = corrTemp.T[w][::-1]                                                 #1 
            # get the values in uu reverse by vv
            rankIndex = uu[s][::-1]                                                      #1
            # Create a dictionairy and assign values to it.
            model= {}                                                                    #1
            model['featureIndex'] = rankIndex                                            #1
            model['rankValue'] = values                                                  #1
            model['featureRankingMethod'] = 'fishersMultiClassFeatureRanking'            #1
            
            return model                                                                 #1

if __name__ == '__main__':
    '''
    setup to use the iris data set as an example.


    '''
    # Open the iris data set
    df = pd.read_csv("../data/iris.csv")
    df['cat'] =pd.factorize(df['species'])[0]
    print(df.head())
    y = df[['species','cat']]
    x = df.drop(['species','cat'],axis=1).values

    print("x.shape {}".format(x.shape))
    print("y.shape {}".format(y.shape))
    print("end")
