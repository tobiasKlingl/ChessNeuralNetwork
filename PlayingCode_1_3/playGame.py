import numpy as np
import BoardPositions as bp
import functions
from timeit import default_timer as timer
import multiprocessing as mp
import pickle
import os
import random
import copy
import sys
sys.path.append("../NetworkCode/")
import NNlib_1_6 as nn

debug=False
noOutputMode=False#True
printMergeInfo=True
colored=False
useMultiprocessing=True#False

#neurons:[input, --hidden-- ,output] layer
sizes    =[780  ,1500,250,150,    1]
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
    checkForDoubles=int(argin[6])
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
    net=nn.load(mainDir+subDir+Inputname)
else:
    net=""
    
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
    for i,(ID,win) in enumerate(zip(gamePositionIDs,winners)):
        alreadyIncluded=False
        for j,(mergedID,mergedWin,mergedCount) in enumerate(zip(mergedPositionIDs,mergedWinners,mergedCounter)):
            if ID==mergedID:
                mergedWinners[j],mergedCounter[j]=mergeStuff(i,j,ID,win,1,mergedWin,mergedCount)
                alreadyIncluded=True
                break
        if(len(mergedPositionIDs)==0 or alreadyIncluded==False):
            if(printMergeInfo): print("INFO: Adding new position ("+str(ID)+") at position",str(len(mergedPositionIDs))+". winners["+str(i)+"]=",win)
            mergedPositions.append(gamePositions[i])
            mergedWinners.append(win)
            mergedCounter.append(1)
            mergedPositionIDs.append(ID)
    merged_data=[mergedPositions,mergedWinners,mergedPositionIDs,mergedCounter]
    return merged_data

def update_data(i,data,mergedPos,mergedOutput,mergedID,mergedCounter):
    if(printMergeInfo): print("Update output file!")
    allPos,allOutput,allIDs,allCounter=data[0],data[1],data[2],data[3]
    lenAllIDs=len(allIDs)
    alreadyIncluded=False
    if(lenAllIDs>0):
        for j,(IDAll,countAll,outAll) in enumerate(zip(allIDs,allCounter,allOutput)):
            if mergedID==IDAll:
                allOutput[j],allCounter[j]=mergeStuff(i,j,IDAll,mergedOutput,mergedCounter,outAll,countAll)
                alreadyIncluded=True
                break
    if(lenAllIDs==0 or alreadyIncluded==False):
        lenAllIDs=len(allIDs)
        if plyToStart<=checkForDoubles or len(mergedID)<25:
            if(printMergeInfo):
                if plyToStart<=checkForDoubles:
                    if(printMergeInfo): print("plyToStart=",plyToStart,"<=",checkForDoubles,"=checkForDoubles!")
                elif len(mergedID)<25:
                    if(printMergeInfo): print("len(mergedID)=",len(mergedID),"<25!")
            if(printMergeInfo): print("INFO: Adding new position (",mergedID,") at position",str(lenAllIDs)+". mergedOutput=",mergedOutput)
            allPos.insert(    lenAllIDs,mergedPos    )
            allOutput.insert( lenAllIDs,mergedOutput )
            allCounter.insert(lenAllIDs,mergedCounter)
            allIDs.insert(    lenAllIDs,mergedID     )
        else:
            if(printMergeInfo): print("INFO: Adding new position (",mergedID,") at position",str(len(allPos))+". mergedOutput=",mergedOutput,". Not added to allIDs & allCounter!")
            allPos.append(mergedPos)
            allOutput.append(mergedOutput)
        data=[allPos,allOutput,allIDs,allCounter]
    return data

def updateOutput(train, valid, test, mdata):
    mergedPos,mergedOutput,mergedIDs,mergedCounter=mdata[0],mdata[1],mdata[2],mdata[3]
    for i,(pos,out,ID,count) in enumerate(zip(mergedPos,mergedOutput,mergedIDs,mergedCounter)):
        randm=random.randint(1,100)
        if(printMergeInfo): print("\nINFO: Adding to Training!")
        train=update_data(i,train,pos,out,ID,count)
        if(randm<10):
            if(printMergeInfo): print("\nINFO: Adding to Validation (randm="+str(randm)+")!")
            valid=update_data(i,valid,pos,out,ID,count)
        #else:
        #    if(printMergeInfo): print("INFO: Adding to Test")
        #    test=update_data(i,test,pos,out,ID,count)
    return train,valid,test

def play(idx,boardState):
    start_timeit = timer()
    idxString=str(idx)
    if(boardState.Finished==True):
        return "","",-1,boardState.Finished
    print(idxString+": ### Playing game number",idx,"###")
    initialPlyNumber=boardState.PlyNumber
    initialPlayer,initialOpponent=boardState.CurrentPlayer,boardState.CurrentOpponent
    print(idxString+": INFO: initialPlyNumber,initialPlayer,initialOpponent=",initialPlyNumber,initialPlayer,initialOpponent)
    player=boardState.CurrentPlayer
    while(boardState.Finished==False):
        if(noOutputMode==False):
            if colored:print("\n"+idxString+": \033[1;31;48m###### Ply ",boardState.PlyNumber," (player:",str(boardState.CurrentPlayer)+") ######\033[1;37;48m")
            else:      print("\n"+idxString+": ###### Ply "             ,boardState.PlyNumber," (player:",str(boardState.CurrentPlayer)+") ######")
        #start_time_nextMove = timer()
        boardState.Finished=functions.nextMove(net,boardState,colored,debug,noOutputMode)
        #print("time for nextMove",timer()-start_time_nextMove)
        if(boardState.PlyNumber==initialPlyNumber):
            finishedInOne=boardState.Finished
            boardInput=boardState.getInput()
            boardInputID=boardState.getInputID(debug)
        if(boardState.PlyNumber>=StopGameNumber):
            winner=boardState.setWinner(0.5)
            if(noOutputMode==False): print(idxString+": INFO: More than",StopGameNumber,"plys played. This is a draw.")
            break
        else:
            boardState.nextPly()
    winner=boardState.getWinner()
    textColor,resetColor="",""
    if(colored==True): textColor,resetColor="\033[1;31;49m","\033[1;37;49m"
    if(winner==initialPlayer):
        won=1
        print(textColor+idxString+": Player",winner,"won this game!!! in",boardState.PlyNumber,"plys.",resetColor)
    elif(winner==initialOpponent):
        won=0
        print(textColor+idxString+": Player",winner,"won this game!!! in",boardState.PlyNumber,"plys.",resetColor)
    elif(winner==0.5):
        won=0.5
        print(textColor+idxString+": Game ended remis!!!",resetColor)
    else:
        print(idxString+": ERROR: Unknown value for winner! winner=",winner)
    print(idxString+": Time for game number ("+idxString+") =",timer()-start_timeit)
    return boardInput,boardInputID,won,finishedInOne
        
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
        boardpositions.reconstructGameState(pieceBoardPositions[0],plyToStart)
        boardpositions.nextPly()
    else:
        boardpositions.initializeBoard(colored,debug,noOutputMode)
    start_totalTime = timer()
    gamePositions,gamePositionIDs,winners=[],[],[]
    exitState=0
    if useMultiprocessing:
        iterable=[copy.deepcopy(boardpositions) for i in range(numberOfGamesToPlay)]
        pool=mp.Pool()
        count=0
        result=pool.starmap_async(play,enumerate(iterable))
        for gamePos,gameID,winner,finishedInOne in result.get():
            if winner!=-1:
                gamePositions.append(gamePos)
                winners.append(winner)
                gamePositionIDs.append(gameID)
            count+=1
            if finishedInOne==True:
                exitState=1
                print("Game finished => exitState=",exitState)
        pool.close()
        pool.join()
    else:
        for idx in range(numberOfGamesToPlay):
            copiedBoardPositions=copy.deepcopy(boardpositions)
            gamePos,gameID,winner,finishedInOne=play(idx,copiedBoardPositions)
            if winner!=-1:
                if printMergeInfo: print("Appending (counter="+str(idx)+") gameID,winner=",gameID,winner,"to lists.")
                gamePositions.append(gamePos)
                gamePositionIDs.append(gameID)
                winners.append(winner)
            else:
                if printMergeInfo: print("Nothing to append! (counter="+str(idx)+") Winner is",winner)
            if finishedInOne==True:
                exitState=1
                print("Game finished => exitState=",exitState)
    print("\nAll games starting at ply",plyToStart,"are finished! Collect the data.\n")
    print("Total time for all games starting at ply",plyToStart,":",timer()-start_totalTime)
    start_timeit = timer()
    print("\nNow merge data of ply",plyToStart)
    merged_data=mergePositions(gamePositions, gamePositionIDs, winners)
    print("\nNow merge the data with the data from previous plys!")
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
        printMergeInfo: print("Write max element",i,"to file:")
        maxPosition=positions.pop(maxIndex)
        maxVal=winners.pop(maxIndex)
        ID=IDs.pop(maxIndex)
        print("maxVal=",maxVal,"; ID=",ID)
        posToWrite.append(maxPosition)
    writeBestPos = open("data/bestLastMoves.pkl", "wb")
    pickle.dump(posToWrite,writeBestPos)
    print("Time for writing all the stuff to file=",timer()-start_timeit)
    if exitState==1:
        sys.exit(1)
else:
    play(0,boardpositions)
