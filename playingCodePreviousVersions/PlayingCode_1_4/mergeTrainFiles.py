import pickle
from otherStuff import updateOutput
from timeit import default_timer as timer
import sys
import os
import glob

argin=sys.argv
try:
    it=int(argin[1])          #neural network iteration id. it=0 => random vs. random
    mergeInfo=bool(argin[2])
except:
    it=0
    mergeInfo=True
    pass

Dir="data/"
outFile="training_data_iteration_"+str(it)+".pkl"
baseFileName="training_data_iteration_"+str(it)

def saveFile(trainFile,train_data,valid_data,test_data):
    fileOut = open(trainFile, "wb")
    pickle.dump([train_data,valid_data,test_data],fileOut)
    fileOut.close()

def loadGameFile(gameFile):
    fileIn = open(gameFile,"rb")
    game_data = pickle.load(fileIn)
    return game_data

### Start here
start_merge=timer()
if os.path.exists(Dir+outFile):
    print("INFO:",Dir+outFile,"already exists. Import!")
    fileIn = open(Dir+outFile,"rb")
    train_data, valid_data, test_data = pickle.load(fileIn)
    fileIn.close()
else:
    print("INFO: No",Dir+outFile,"yet. Nothing to import!")
    train_data, valid_data, test_data=[[],[],[],[]], [[],[],[],[]], [[],[],[],[]]
print("INFO: before len(allTrainPos/IDs/Winners/Counter)=",len(train_data[0]),"/",len(train_data[2]),"/",len(train_data[1]),"/",len(train_data[3]))
print("INFO: before len(allValidPos/IDs/Winners/Counter)=",len(valid_data[0]),"/",len(valid_data[2]),"/",len(valid_data[1]),"/",len(valid_data[3]))
print("INFO: before len(allTestPos/IDs/Winners/Counter)=" ,len(test_data[0]) ,"/",len(test_data[2]) ,"/",len(test_data[1]) ,"/",len(test_data[3]))    

for job,filename in enumerate(sorted(glob.iglob(Dir+baseFileName+"_*"))):
    print("Loading game data:",filename)
    game_data=loadGameFile(filename)
    
    print("\nNow merge the data from job",job,"with the data from other jobs!")
    train_data,valid_data,test_data=updateOutput(train_data, valid_data, test_data, game_data, mergeInfo)
    print("\nStuff merged!")
    print("INFO: after len(allTrainPos/IDs/Out/Counter)=",len(train_data[0]),"/",len(train_data[2]),"/",len(train_data[1]),"/",len(train_data[3]))
    print("INFO: after len(allValidPos/IDs/Out/Counter)=",len(valid_data[0]),"/",len(valid_data[2]),"/",len(valid_data[1]),"/",len(valid_data[3]))
    print("INFO: after len(allTestPos/IDs/Out/Counter)=" ,len(test_data[0]) ,"/",len(test_data[2]) ,"/",len(test_data[1]) ,"/",len(test_data[3])) 

saveFile(Dir+outFile,train_data,valid_data,test_data)
print("Time for final merge:",timer()-start_merge)

