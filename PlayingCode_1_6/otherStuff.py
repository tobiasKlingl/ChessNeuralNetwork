import pickle
import random
import numpy as np

def mergeStuff(i,j,ID,winner,counter,mergedWinner,mergedCounter,mergeInfo):
    if(mergeInfo):
        print("=> Position (",ID,") has been played",np.sum(mergedCounter),"times before (position",str(j)+").")
        print("=> winners["+str(i)+"]=",winner,", mergedWinners[" +str(j)+"]=",mergedWinner,", mergedCounter["+str(j)+"]=",mergedCounter)
    mergedWinner=np.nan_to_num(np.divide(np.multiply(winner,counter)+np.multiply(mergedWinner,mergedCounter),counter+mergedCounter))
    mergedCounter+=counter
    if(mergeInfo): print("=> New mergedWinners["+str(j)+"]=",mergedWinner,", mergedCounter["+str(j)+"]=",mergedCounter)
    return mergedWinner,mergedCounter

def update_data(i,data,gamePos,gameOutput,gameID,gameCounter,mergeInfo):
    if(mergeInfo): print("Update output file!")
    allPos,allOutput,allIDs,allCounter=data[0],data[1],data[2],data[3]
    lenAllIDs =len(allIDs)
    lenGameIDs=len(gameID)
    print("lenAllIDs,lenGameIDs",lenAllIDs,lenGameIDs)
    if lenAllIDs>0:
        if lenGameIDs>0:
            alreadyIncluded=False
            for j,IDAll in enumerate(allIDs):
                if gameID==IDAll:
                    allOutput[j],allCounter[j]=mergeStuff(i,j,gameID,gameOutput,gameCounter,allOutput[j],allCounter[j],mergeInfo)
                    if(mergeInfo):
                        for l in range(1882):
                            print("=> New allOutput["+str(j)+"]["+str(l)+"]=",allOutput[j][l],", new allCounter["+str(j)+"]["+str(l)+"]=",allCounter[j][l])
                    alreadyIncluded=True
                    break
            if alreadyIncluded==False:
                if(mergeInfo): print("Insert new ID ",gameID," at position",str(lenAllIDs)+". Output["+str(lenAllIDs)+"]=",gameOutput,"; Counter["+str(lenAllIDs)+"]=",gameCounter)
                allPos.insert(    lenAllIDs,gamePos    )
                allOutput.insert( lenAllIDs,gameOutput )
                allIDs.insert(    lenAllIDs,gameID     )
                allCounter.insert(lenAllIDs,gameCounter)
        else:
            if(mergeInfo): print("Append new position at position",str(len(allPos))+". Output["+str(len(allOutput))+"]=",gameOutput)
            allPos.append(gamePos)
            allOutput.append(gameOutput)
    else:
        if(mergeInfo): print("INFO: Add first ID",gameID,"at position",str(lenAllIDs)+". Output=",gameOutput)
        allPos.append(gamePos)
        allOutput.append(gameOutput)
        if len(gameID)>0:
            allIDs.append(gameID)
            allCounter.append(gameCounter)
    data=[allPos,allOutput,allIDs,allCounter]
    return data

def updateOutput(train, valid, test, gdata, mergeInfo):
    gamePos,gameOutput,gameIDs,gameCounters=gdata[0],gdata[1],gdata[2],gdata[3]
    lenGameCounters=len(gameCounters)
    for i,(pos,out) in enumerate(zip(gamePos,gameOutput)):
        randm=random.randint(1,100)
        if(randm>=10):
            if(mergeInfo): print("\nINFO: Adding to Training!")
            if i<lenGameCounters:
                train=update_data(i,train,pos,out,gameIDs[i],gameCounters[i],mergeInfo)  # check move for doubles
            else:
                #c=-1*np.ones(1882,dtype=np.int64)
                train=update_data(i,train,pos,out,"",-1,mergeInfo) # dont check move for doubles
        if(randm<10):
            if(mergeInfo): print("\nINFO: Adding to Validation (randm="+str(randm)+")!")
            if i<lenGameCounters:
                valid=update_data(i,valid,pos,out,gameIDs[i],gameCounters[i],mergeInfo)
            else:
                #c=-1*np.ones(1882,dtype=np.int64)
                valid=update_data(i,valid,pos,out,"",-1,mergeInfo)
        """
        else:
            if(mergeInfo): print("INFO: Adding to Test")
            if i<lenGameCounters:
                test=update_data(i,test,pos,out,gameIDs[i],gameCounters[i],mergeInfo)
            else:
                test=update_data(i,test,pos,out,"",-1,mergeInfo)
        """
    return train,valid,test

def loadBestPos(bestPosFile):
    bestPosToStart = open(bestPosFile, "rb")
    return pickle.load(bestPosToStart)

def writeBestPosToFile(posToWrite,bestPosFile):
    writeBestPos = open(bestPosFile, "wb")
    pickle.dump(posToWrite,writeBestPos)
    writeBestPos.close()
