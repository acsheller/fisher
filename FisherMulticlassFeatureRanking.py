'''
This is a cleaner model of Fisher Multcast Feature Ranking.



'''

import numpy as np
import pandas as pd
import sys
# If there are alot of observations this mprevents warnings from being presented.
np.seterr(divide='ignore', invalid='ignore')
class FisherMultiClassFeatureRanker(object):
    '''
    Class for Fisher Multi-Class Feature Ranking
    '''

    def fmcfr(self,X=None,Y=None,method=1):
        '''
        An Implementation of Fisher Multi-Class Feature Ranking
        That doesn't know how many features
        '''
        # How many features minus the label
        vDim  = len(X.columns)
        
        # The number of unique labels in our data
        oDim = len(Y.unique())
        
        # Create a label repository that is oDim (unique labels) in size
        # Store the index values of the observations  of each
        # label in a dictionairy
        labelDict = {}
        for label in Y.unique():
            labelDict[label] = Y[Y==label].index.values
        #setosaLabels = np.nonzero(Y.values == 'setosa')                                #1
        #versicolorLabels  = np.nonzero(Y.values == 'versicolor')                       #1
        #virginicaLabels = np.nonzero(Y.values == 'virginica')                          #1
        
        # Now get the standard deviation of the datafrme
        STDss = []                                                                      #1
        MEANss = []  
        STDs = []
        MEANs = []                                                                   #1
        if oDim < 2: # if two-class # not doing this yet
            print("Sorry not yet implemented")                                         #1
        
        # We have more than two classes
        else:
            # Get Standard Deviations for each species seperately
            # And then add back to the array

            for key in labelDict.keys():
                if len(STDss) == 0:
                    STDss = X.loc[labelDict[key]].std().values
                    MEANss = np.array(X.loc[labelDict[key]].mean().values)
                else:
                    STDss = np.add(STDss,X.loc[labelDict[key]].std().values)
                    MEANss = (np.concatenate((MEANss,X.loc[labelDict[key]].mean().values)))

            MEANss = MEANss.reshape(oDim,vDim)
            #MS is Mean Over Standard Deviation
            
            MS = MEANss/STDss
            
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
    y = df['species']
    x = df.drop('species',axis=1)

    fisherRanker = FisherMultiClassFeatureRanker()
    print("Performing Ranking on Iris Data Set.")
    model = fisherRanker.fmcfr(x,y)
    print(model)
    print("Ranking on Iris Data Set Complete.")
    print("\n")

    df = pd.read_csv("../data/train.csv")
    y = df['label']
    x = df.drop('label',axis=1)
    # One should pre-process the mnist data set in someway
    print("performing Ranking on  unproessed MNIST data set.")
    model2 = fisherRanker.fmcfr(x,y)
    print(model2['featureIndex'][:150,])
    print("Ranking on unporcessed MNIST data set complete.\n")

    # Now need to do with the 42000 processed data set. 
    df = pd.read_excel(r"../data/trainFeatures42k.xls",header=None)
    df = df.rename(columns={0:'label'})
    y = df['label']
    x = df.drop('label',axis=1)
    print("feature Ranking on processed data new way")
    model3= fisherRanker.fmcfr(x,y)
    print(model3['featureIndex'][:150,])
    print("Ranking on porcessed MNIST data set complete.")

    # Now I need to do the 2 features example for Iris
    df = pd.read_csv('../data/iris.csv')
