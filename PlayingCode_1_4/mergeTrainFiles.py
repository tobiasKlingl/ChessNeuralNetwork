import pickle
from otherStuff import updateOutput
from timeit import default_timer as timer
import sys
import os

argin=sys.argv
try:
    numJobs=int(argin[1])
    it=int(argin[2])              #neural network iteration id. it=0 => random vs. random
    mergeInfo=bool(argin[3])
except:
    numJobs=8
    it=1
    mergeInfo=False
    pass

outFile="data/training_data_iteration_"+str(it)+".pkl"
baseFileName="data/training_data_iteration_"+str(it)

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
if os.path.exists(outFile):
    print("INFO:",outFile,"already exists. Import!")
    fileIn = open(outFile,"rb")
    train_data, valid_data, test_data = pickle.load(fileIn)
    fileIn.close()
else:
    print("INFO: No",outFile,"yet. Nothing to import!")
    train_data, valid_data, test_data=[[],[],[],[]], [[],[],[],[]], [[],[],[],[]]
print("INFO: before len(allTrainPos/IDs/Winners/Counter)=",len(train_data[0]),"/",len(train_data[2]),"/",len(train_data[1]),"/",len(train_data[3]))
print("INFO: before len(allValidPos/IDs/Winners/Counter)=",len(valid_data[0]),"/",len(valid_data[2]),"/",len(valid_data[1]),"/",len(valid_data[3]))
print("INFO: before len(allTestPos/IDs/Winners/Counter)=" ,len(test_data[0]) ,"/",len(test_data[2]) ,"/",len(test_data[1]) ,"/",len(test_data[3]))    

for jobNum in range(numJobs):
    print("Loading game data:",baseFileName+"_"+str(jobNum)+".pkl")
    game_data=loadGameFile(baseFileName+"_"+str(jobNum)+".pkl")
    
    print("\nNow merge the data from job",jobNum,"with the data from other jobs!")
    train_data,valid_data,test_data=updateOutput(train_data, valid_data, test_data, game_data, mergeInfo)
    print("\nStuff merged!")
    print("INFO: after len(allTrainPos/IDs/Out/Counter)=",len(train_data[0]),"/",len(train_data[2]),"/",len(train_data[1]),"/",len(train_data[3]))
    print("INFO: after len(allValidPos/IDs/Out/Counter)=",len(valid_data[0]),"/",len(valid_data[2]),"/",len(valid_data[1]),"/",len(valid_data[3]))
    print("INFO: after len(allTestPos/IDs/Out/Counter)=" ,len(test_data[0]) ,"/",len(test_data[2]) ,"/",len(test_data[1]) ,"/",len(test_data[3])) 

saveFile(outFile,train_data,valid_data,test_data)
print("Time for final merge:",timer()-start_merge)
