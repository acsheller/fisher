'''
This module is intended to perform fisher multicast feature ranking.



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


