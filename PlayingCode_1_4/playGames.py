import numpy as np
import BoardPositions as bp
import functions
from otherStuff import update_data,mergeStuff,writeBestPosToFile,loadBestPos
from timeit import default_timer as timer
import pickle
import json
import os
import random
import sys
import numba as nb

Debug=False#True
NoOutput=False#True
PrintMergeInfo=True
Col=False

Inputname="NADAM_L2_maxEpochs_500_nEpochs_20_EtaRed_20_mBatchSize_100_eta_0.1_lmbda_1.5_mu_0.2_relu"
#sizes=nb.typed.List((780,500,250,50,1))

gameMode="SelfplayRandomVsRandom"
#gameMode="SelfplayNetworkVsRandom"
#gameMode="SelfplayNetworkVsNetwork"

argin=sys.argv
try:
    nGames=int(argin[1])
    StopGameNumber=int(argin[2])
    upToPly=int(argin[3])
    checkForDoubles=int(argin[4])
    jobNum=argin[5]
    iteration=int(argin[6])
    mergeInfo=bool(argin[7])
except:
    nGames=100
    StopGameNumber=500
    upToPly=500
    checkForDoubles=5
    jobNum="1"
    iteration=1
    mergeInfo=False
    pass

boardpositions=nb.typed.List([bp.BoardPositions() for i in range(nGames)])
gameDataFile="data/training_data_iteration_"+str(iteration)+"_"+jobNum+".pkl"

itMinus1=iteration-1
if(gameMode=="SelfplayNetworkVsRandom" or gameMode=="SelfplayNetworkVsNetwork"):
    mainDir="/home/tobias/ChessNeuralNetwork/NetworkCode/saves/"
    subDir ="iteration_"+str(itMinus1)+"/"
    f = open(mainDir+subDir+Inputname, "r")
    data = json.load(f)
    f.close()
    #Sizes  =sizes
    Sizes  =nb.typed.List(data["sizes"])
    Weights=nb.typed.List([np.array(w) for w in data["weights"]])
    Biases =nb.typed.List([np.array(b) for b in data["biases"]])
else:
    Sizes  =nb.typed.List((780,3,1))  # <- just dummy network
    Weights=nb.typed.List([np.zeros((y,x),dtype=np.float64) for y,x in zip(Sizes[1:],Sizes[:-1])])
    Biases =nb.typed.List([np.zeros((y,1),dtype=np.float64) for y   in Sizes[1:]])


@nb.njit
def play(idxString,Sizes,Weights,Biases,boardState,debug,noOutput,col):
    if(boardState.Finished==True):
        return np.zeros(780,dtype=np.int64),"",-9.,boardState.Finished
    finishedInOne=False
    if noOutput==False: print(idxString+": ### Playing game number",idxString,"###")
    initialPlyNumber=boardState.PlyNumber
    initialPlayer,initialOpponent=boardState.CurrentPlayer,boardState.CurrentOpponent
    if noOutput==False: print(idxString+": INFO: initialPlyNumber,initialPlayer,initialOpponent=",initialPlyNumber,initialPlayer,initialOpponent)
    player=boardState.CurrentPlayer
    while(boardState.Finished==False):
        if(noOutput==False):
            if col:print("\n"+idxString+": \033[1;31;48m###### Ply ",boardState.PlyNumber," (player:",str(boardState.CurrentPlayer)+") ######\033[1;37;48m")
            else:  print("\n"+idxString+": ###### Ply "             ,boardState.PlyNumber," (player:",str(boardState.CurrentPlayer)+") ######")
        boardState.Finished=functions.nextMove(boardState,Sizes,Weights,Biases,col,debug,noOutput)
        if(boardState.PlyNumber==initialPlyNumber):
            finishedInOne=boardState.Finished
            boardInput=boardState.getInput()
            boardInputID=boardState.getInputID(False)
        if(boardState.PlyNumber>=StopGameNumber):
            winner=boardState.setWinner(0)
            if(noOutput==False): print(idxString+": INFO: More than",StopGameNumber,"plys played. This is a draw.")
            break
        else:
            boardState.nextPly()
    winner=boardState.getWinner()
    textColor,resetColor="",""
    if(col==True): textColor,resetColor="\033[1;31;49m","\033[1;37;49m"
    if(winner==initialPlayer):
        won=1.
        #if noOutput==False:
        print(textColor+idxString+": Player",winner,"won in",boardState.PlyNumber,"plys!",resetColor)
    elif(winner==initialOpponent):
        won=0.
        #if noOutput==False:
        print(textColor+idxString+": Player",winner,"won in",boardState.PlyNumber,"plys!",resetColor)
    elif(winner==0):
        won=0.5
        #if noOutput==False:
        print(textColor+idxString+": Game ended remis!",resetColor)
    else:
        won=-9.
        print(idxString+": ERROR: Unknown value for winner! winner=",winner)
    return boardInput,boardInputID,won,finishedInOne

def playTheGames(startPly,Sizes,Weights,Biases,boardpositions,deb,noOut,mergeInfo,c):
    N=len(boardpositions)
    gamePositions,gamePositionIDs,gameWinners,gameCounters=[],[],[],[]
    exitState=0
    for idx in range(N):
        idxString=str(idx)
        boardposition=boardpositions[idx]
        gamePos,gameID,gameWinner,finishedInOne=play(idxString,Sizes,Weights,Biases,boardposition,deb,noOut,c)
        if mergeInfo: print(idxString+": Got game",idx,": gameID,winner=",gameID,gameWinner)
        lenCounters=len(gameCounters)
        if lenCounters>0:
            alreadyIncluded=False
            for j,ID in enumerate(gamePositionIDs):
                if ID==gameID:
                    gameWinners[j],gameCounters[j]=mergeStuff(idx,j,gameID,gameWinner,1,gameWinners[j],gameCounters[j],mergeInfo)
                    alreadyIncluded=True
                    break
            if(alreadyIncluded==False):
                if(mergeInfo): print(idxString+": Insert new ID at position",str(lenCounters)+". gameWinners["+str(lenCounters)+"]=",gameWinner)
                gamePositions.insert(  lenCounters,gamePos)
                gameWinners.insert(    lenCounters,gameWinner)
                gameCounters.insert(   lenCounters,1)
                gamePositionIDs.insert(lenCounters,gameID)
        else:
            if(mergeInfo): print(idxString+": Append new ID. gameWinners["+str(len(gameWinners))+"]=",gameWinner)
            gamePositions.append(gamePos)
            gameWinners.append(gameWinner)
            gameCounters.append(1)
            gamePositionIDs.append(gameID)
        if finishedInOne==True:
            exitState=1
            print(idxString+": Games finished => exitState=",exitState)
            break
    return gamePositions,gamePositionIDs,gameWinners,gameCounters,exitState

def runSelfPlay(boardpositions,it,jobNum,startPly,nGames,Sizes,Weights,Biases,deb,noOut,mergeInfo,c):
    print("\nPlaying from next position. startPly=",startPly)
    bestPosFile="data/bestLastMoves_"+jobNum+".pkl"
    if(startPly>0):
        pieceBoardPositions = loadBestPos(bestPosFile)
        for i,bp in enumerate(boardpositions):
            bp.GameMode=gameMode
            bp.reconstructGameState(i,pieceBoardPositions[0].astype(np.int64),startPly)
            bp.nextPly()
    else:
        for i,bp in enumerate(boardpositions):
            bp.GameMode=gameMode
            bp.initializeBoard(i,c,deb,noOut)
    gamePositions,gamePositionIDs,gameWinners,gameCounters,exitState=playTheGames(startPly,Sizes,Weights,Biases,boardpositions,deb,noOut,mergeInfo,c)
    print("\nAll games starting at ply",startPly,"(jobNum="+jobNum+") are finished! Write the data to file.\n")
    
    #Update gameDataFile
    if os.path.exists(gameDataFile):
        print("INFO:",gameDataFile,"already exists. Import!")
        fileIn = open(gameDataFile,"rb")
        merged_data = pickle.load(fileIn)
        fileIn.close()
        for i,(pos,out,ID,count) in enumerate(zip(gamePositions,gameWinners,gamePositionIDs,gameCounters)):
            if startPly<checkForDoubles or len(ID)<25:
                merged_data=update_data(i,merged_data,pos,out,ID,count,mergeInfo)
            else:
                merged_data[0].append(pos)
                merged_data[1].append(out)
    else:
        print("INFO: No",gameDataFile,"yet. Nothing to import!")
        merged_data=[gamePositions,gameWinners,gamePositionIDs,gameCounters]
    fileOut = open(gameDataFile, "wb")
    pickle.dump(merged_data,fileOut)
    fileOut.close()
          
    #Write the most promising position to file
    posToWrite=[]
    idWinPerc=[(ID,round(num,4)) for ID,num in zip(gamePositionIDs,gameWinners)]
    for idWin in idWinPerc:
        print("ID (win percentage)=",idWin[0]+": ("+str(idWin[1])+")")
    for i in range(1):
        maxIndex=gameWinners.index(max(gameWinners))
        if mergeInfo: print("Write max element",i,"to file:")
        maxPosition=gamePositions[maxIndex]
        maxVal=gameWinners[maxIndex]
        ID=gamePositionIDs[maxIndex]
        #maxPosition=gamePositions.pop(maxIndex)
        #maxVal=gameWinners.pop(maxIndex)
        #ID=gamePositionIDs.pop(maxIndex)
        print("Writing ID=",ID,"with maxVal=",maxVal,"to file.")
        posToWrite.append(maxPosition)
    writeBestPosToFile(posToWrite,bestPosFile)
    print("INFO: len(gamePositions/Winners/IDs/Counters)=",len(merged_data[0]),"/",len(merged_data[1]),"/",len(merged_data[2]),"/",len(merged_data[3]))
    return exitState
        
### Start the code        
if("Selfplay" in gameMode):
    exitState=0
    print("Iteration=",iteration)
    for startPly in range(upToPly):
        if exitState==0:
            start_timeit = timer()
            exitState=runSelfPlay(boardpositions,iteration,jobNum,startPly,nGames,Sizes,Weights,Biases,Debug,NoOutput,PrintMergeInfo,Col)
            print("Time for",nGames,"games (startPly=",startPly,"):",timer()-start_timeit)
        else:
            print("exitState =",exitState,"! Breaking loop!")
            break
else:
    play(0,Sizes,Weights,Biases,boardpositions,debug,noOutput,col)
