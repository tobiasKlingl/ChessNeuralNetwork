import numpy as np
import BoardPositions as bp
import functions
#import time
from timeit import default_timer as timer
import multiprocessing as mp
import pickle
import os
import random
import copy
import sys
sys.path.append("../../NetworkCode/NetworkCodePreviousVersions/NN_1_4/")
import NNlib_1_4 as nn

debug=False
noOutputMode=False#True
printMergeInfo=True
colored=False
useMultiprocessing=False#True

sizes=[780  ,1500,250,150,    1] #neurons in [input  ,hidden,  output] layer
Inputname="NN_SMGD_L2_maxEpochs_500_nEpochs_20_numEtaRed_10_miniBatchSize_100_etaInit_0.5_lmbda_0.5_mu_0.2"
gameMode="SelfplayRandomVsRandom"
#gameMode="SelfplayNetworkVsRandom"
#gameMode="SelfplayNetworkVsNetwork"
#gameMode="VsComputer"
#gameMode="WhiteVsBlack"

argin=sys.argv
try:
    numberOfGamesToPlay=int(argin[1])
    StopGameNumber=int(argin[2])
    gameMode=argin[3]
    iteration=int(argin[4])
    plyToStart=int(argin[5])
except:
    numberOfGamesToPlay=1
    StopGameNumber=200
    pass

boardpositions=bp.BoardPositions()
boardpositions.setGameMode(gameMode)

iterationMinus1=iteration-1
if(gameMode=="SelfplayNetworkVsRandom" or gameMode=="SelfplayNetworkVsNetwork"):
    mainDir="/home/tobias/MachineLearning/Chess/NetworkCode/saves/"
    subDir ="CrossEntropy/iteration_"+str(iterationMinus1)+"/"
    print("Loading Network:",mainDir+subDir+Inputname)
    net = nn.load(mainDir+subDir+Inputname)

def mergeStuff(i,j,ID,winner,counter,mergedWinner,mergedCounter):
    if(printMergeInfo):
        print("INFO: This position (",ID,") has been played",mergedCounter,"times before (position",str(j)+").")
        print("winners["+str(i)+"]=",winner,", mergedWinners[" +str(j)+"]=",mergedWinner)
        print("mergedCounter["+str(j)+"]=",mergedCounter)
    mergedWinner=(winner*counter+mergedWinner*mergedCounter)/(counter+mergedCounter)
    mergedCounter+=counter
    if(printMergeInfo): print("New mergedWinners["+str(j)+"]=",mergedWinner,", mergedCounter["+str(j)+"]=",mergedCounter)
    return mergedWinner,mergedCounter
    
def mergePositions(gamePositions, gamePositionIDs, winners):
    if(printMergeInfo): print("INFO: Merge played positions!")
    mergedPositions,mergedPositionIDs,mergedWinners,mergedCounter=[],[],[],[]
    for i,ID in enumerate(gamePositionIDs):
        alreadyIncluded=False
        for j,mergedID in enumerate(mergedPositionIDs):
            if ID==mergedID:
                mergedWinners[j],mergedCounter[j]=mergeStuff(i,j,ID,winners[i],1,mergedWinners[j],mergedCounter[j])
                alreadyIncluded=True
                break
        if(len(mergedPositionIDs)==0 or alreadyIncluded==False):
            if(printMergeInfo): print("INFO: Adding new position ("+str(ID)+") at position",str(len(mergedPositionIDs))+". winners["+str(i)+"]=",winners[i])
            mergedPositions.append(gamePositions[i])
            mergedWinners.append(winners[i])
            mergedCounter.append(1)
            mergedPositionIDs.append(ID)
    merged_data=[mergedPositions,mergedWinners,mergedPositionIDs,mergedCounter]
    return merged_data

def update_data(i,data,mergedPos,mergedOutput,mergedID,mergedCounter):
    if(printMergeInfo): print("Update output file!")
    allPos,allOutput,allIDs,allCounter=data[0],data[1],data[2],data[3]
    alreadyIncluded=False
    if(len(allIDs)>0):
        for j,IDAll in enumerate(allIDs):
            if mergedID==IDAll:
                allOutput[j],allCounter[j]=mergeStuff(i,j,IDAll,mergedOutput,mergedCounter,allOutput[j],allCounter[j])
                alreadyIncluded=True
                break
    if(len(allIDs)==0 or alreadyIncluded==False):
        if(printMergeInfo): print("INFO: Adding new position (",mergedID,") at position",str(len(allIDs))+". mergedOutput=",mergedOutput)
        allPos.append(mergedPos)
        allOutput.append(mergedOutput)
        allCounter.append(mergedCounter)
        allIDs.append(mergedID)
    data=[allPos,allOutput,allIDs,allCounter]
    return data

def updateOutput(train, valid, test, mdata):
    mergedPos,mergedOutput,mergedIDs,mergedCounter=mdata[0],mdata[1],mdata[2],mdata[3]
    for i,ID in enumerate(mergedIDs):
        randm=random.randint(1,100)
        if(printMergeInfo): print("\nINFO: Adding to Training!")
        train=update_data(i,train,mergedPos[i],mergedOutput[i],ID,mergedCounter[i])
        if(randm<10):
            if(printMergeInfo): print("\nINFO: Adding also to Validation (randm="+str(randm)+")!")
            valid=update_data(i,valid,mergedPos[i],mergedOutput[i],ID,mergedCounter[i])
        #else:
        #    if(printMergeInfo): print("INFO: Adding to Test")
        #    test=update_data(i,test,mergedPos[i],mergedOutput[i],ID,mergedCounter[i])
    return train,valid,test

def play(idx,boardState):
    start_timeit = timer()
    if(boardState.Finished==True):
        return "","",-1
    print(idx,": ### Playing game number",idx,"###")
    initialPlyNumber=boardState.PlyNumber
    initialPlayer,initialOpponent=boardState.CurrentPlayer,boardState.CurrentOpponent
    print(idx,": INFO: initialPlyNumber,initialPlayer,initialOpponent=",initialPlyNumber,initialPlayer,initialOpponent)
    player=boardState.CurrentPlayer
    if("SelfplayNetwork" in gameMode): boardState.setNetwork(net,debug,noOutputMode)
    while(boardState.Finished==False):
        if(noOutputMode==False):
            if colored:print("\nidx: \033[1;31;48m###### Ply ",boardState.PlyNumber," (player:",boardState.CurrentPlayer,") ######\033[1;37;48m")
            else:      print("\nidx: ###### Ply "             ,boardState.PlyNumber," (player:",boardState.CurrentPlayer,") ######")
        #start_time_nextMove = timer()
        boardState.Finished=functions.nextMove(boardState,colored,debug,noOutputMode)
        #print("time for nextMove",timer()-start_time_nextMove)
        if(boardState.PlyNumber==initialPlyNumber):
            boardInput=boardState.getInput(debug,noOutputMode)
            boardInputID=boardState.getInputID(debug,noOutputMode)
        if(boardState.PlyNumber>=StopGameNumber):
            winner=boardState.setWinner(0.5)
            if(noOutputMode==False): print(idx,": INFO: More than",StopGameNumber,"plys played. This is a draw.")
            break
        else:
            boardState.nextPly()
    winner=boardState.getWinner()
    textColor,resetColor="",""
    if(colored==True): textColor,resetColor="\033[1;31;49m","\033[1;37;49m"
    if(winner==initialPlayer):
        won=1
        print(textColor+str(idx)+": Player",winner,"won this game!!! in",boardState.PlyNumber,"plys.",resetColor)
    elif(winner==initialOpponent):
        won=0
        print(textColor+str(idx)+": Player",winner,"won this game!!! in",boardState.PlyNumber,"plys.",resetColor)
    elif(winner==0.5):
        won=0.5
        print(textColor+str(idx)+": Game ended remis!!!",resetColor)
    else:
        print(idx,": ERROR: Unknown value for winner! winner=",winner)
    print(idx,": Time for game number ("+str(idx)+") =",timer()-start_timeit)
    return boardInput,boardInputID,won
        
if(gameMode=="VsComputer"):
    playerColor=int(input("Choose your color (white: 0, black: 1):"))
    print("playerColor=",playerColor)
    if playerColor==0:
        gameMode="WhiteVsComputer"
    elif playerColor==1:
        gameMode="BlackVsComputer"
    else:
        print("ERROR: Unknown playerColor in VsComputer mode.")
        
if("Selfplay" in gameMode):
    print("\nPlaying from next position. plyToStart=",plyToStart)
    trainFile="data/training_data_iteration_"+str(iteration)+".pkl"
    if os.path.exists(trainFile):
        print("INFO:",trainFile,"already exists. Import!")
        fileIn = open(trainFile,"rb")
        train_data, valid_data, test_data = pickle.load(fileIn)
        fileIn.close()
    else:
        print("INFO: No",trainFile,"yet. Nothing to import!")
        train_data, valid_data, test_data=[[],[],[],[]], [[],[],[],[]], [[],[],[],[]]
    print("INFO: len(allTrainPos/IDs/Out/Counter)=",len(train_data[0]),"/",len(train_data[2]),"/",len(train_data[1]),"/",len(train_data[3]))
    print("INFO: len(allValidPos/IDs/Out/Counter)=",len(valid_data[0]),"/",len(valid_data[2]),"/",len(valid_data[1]),"/",len(valid_data[3]))
    print("INFO: len(allTestPos/IDs/Out/Counter)=" ,len(test_data[0]) ,"/",len(test_data[2]) ,"/",len(test_data[1]) ,"/",len(test_data[3]))
    
    if(plyToStart>0):
        bestPosToStart = open("data/bestLastMoves.pkl", "rb")
        pieceBoardPositions = pickle.load(bestPosToStart)
        #print("pieceBoardPositions[0]=",pieceBoardPositions[0])
        boardpositions.reconstructGameState(pieceBoardPositions[0],plyToStart)
        boardpositions.nextPly()
    else:
        boardpositions.initializeBoard(colored,debug,noOutputMode)

    start_totalTime = timer()
    gamePositions,gamePositionIDs,winners=[],[],[]
    if useMultiprocessing:
        gamePos,gameIDs,wins=[],[],[]
        start_time = timer()
        iterable=[copy.deepcopy(boardpositions) for i in range(numberOfGamesToPlay)]
        print("Time to get the boardpositions iterable=",timer()-start_time)
        pool=mp.Pool()
        count=0
        result=pool.starmap_async(play,enumerate(iterable))
        for gamePos,gameID,winner in result.get():#pool.starmap_async(play,enumerate(iterable)):
            if winner!=-1:
                print("Appending (counter="+str(count)+") gameID,winner=",gameID,winner,"to lists.")
                gamePositions.append(gamePos)
                gamePositionIDs.append(gameID)
                winners.append(winner)
            else:
                print("Nothing to append! (counter="+str(count)+") Winner is",winner)
            count+=1
        pool.close()
        pool.join()
        print("gamePositionIDs=",gamePositionIDs)
        print("winners=",winners)
    else:
        for idx in range(numberOfGamesToPlay):
            copiedBoardPositions=copy.deepcopy(boardpositions)
            gamePos,gameID,winner=play(0,copiedBoardPositions)
            if winner!=-1:
                print("Appending (counter="+str(idx)+") gameID,winner=",gameID,winner,"to lists.")
                gamePositions.append(gamePos)
                gamePositionIDs.append(gameID)
                winners.append(winner)
            else:
                print("Nothing to append! (counter="+str(idx)+") Winner is",winner)
    print("Total time for all games in iteration("+str(iteration)+") =",timer()-start_totalTime)
    print("Time to get the data!")
    start_timeit = timer()
    merged_data=mergePositions(gamePositions, gamePositionIDs, winners)
    train_data,valid_data,test_data=updateOutput(train_data, valid_data, test_data, merged_data)
    fileOut = open("data/training_data_iteration_"+str(iteration)+"_new.pkl", "wb")
    pickle.dump([train_data,valid_data,test_data],fileOut)
    fileOut.close()
    
    #Write the most promising position to file
    positions=merged_data[0]
    winners=merged_data[1]
    IDs=merged_data[2]
    posToWrite=[]
    print("winners=",winners)
    print("IDs=",merged_data[2])
    for i in range(1):
        maxIndex=winners.index(max(winners))
        print("Write max element",i,"to file:")
        maxPosition=positions.pop(maxIndex)
        maxVal=winners.pop(maxIndex)
        ID=IDs.pop(maxIndex)
        print("maxVal=",maxVal,"; ID=",ID)
        posToWrite.append(maxPosition)
    writeBestPos = open("data/bestLastMoves.pkl", "wb")
    pickle.dump(posToWrite,writeBestPos)
    print("Time for writing all the stuff to file=",timer()-start_timeit)
else:
    play(0,boardpositions)
