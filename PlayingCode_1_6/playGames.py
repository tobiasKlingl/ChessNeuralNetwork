import numpy as np
import BoardPositions as bp
import moves as mv
import functions
from otherStuff import update_data,mergeStuff,writeBestPosToFile,loadBestPos
from timeit import default_timer as timer
import pickle
import json
import os
import random
import sys
import numba as nb

Debug=False #True
NoOutput=True
PrintMergeInfo=True
Col=False  # colored?

# Neural Network to being to used
Inputname="NADAM_L2_maxEpochs_500_nEpochs_20_EtaRed_20_mBatchSize_100_eta_0.1_lmbda_1.0_mu_0.2_relu"

gameMode="Selfplay_RandomVsRandom"
#gameMode="Selfplay_NetworkVsRandom"
#gameMode="Selfplay_NetworkVsNetwork"

argin=sys.argv

try:
    nGames=int(argin[1])
    searchDepth=int(argin[2])
    upToPly=int(argin[3])
    checkForDoubles=int(argin[4])
    jobNum=argin[5]
    iteration=int(argin[6])
    mergeInfo=bool(argin[7])
except:
    nGames=10
    searchDepth=20
    upToPly=1
    checkForDoubles=5
    jobNum="0"
    iteration=1
    mergeInfo=True
    pass

#boardpositions=nb.typed.List([bp.BoardPositions() for i in range(nGames+1)])
boardposition=bp.BoardPositions()
gameDataFile="data/training_data_iteration_"+str(iteration)+"_"+jobNum+".pkl"

moveDict=nb.typed.Dict.empty(
    key_type  =nb.types.UniTuple(nb.int64, 5),
    value_type=nb.int64)
mL,mD=mv.getAllMoves(moveDict) #moveList,moveDict

itMinus1=iteration-1
if(gameMode=="Selfplay_NetworkVsRandom" or gameMode=="Selfplay_NetworkVsNetwork"):
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
def play(idxString,boardInput,Sizes,Weights,Biases,boardState,mD,debug,noOutput,col):
    if(boardState.Finished==1):
        return -1,-9,boardState.Finished
    
    if noOutput==False:
        print(idxString+": ### Playing game number",idxString,"###")
        
    initialPlyNumber=boardState.PlyNumber
    initialPlayer,initialOpponent=boardState.CurrentPlayer,boardState.CurrentOpponent

    if noOutput==False:
        print(idxString+": INFO: initialPlyNumber,initialPlayer,initialOpponent=",initialPlyNumber,initialPlayer,initialOpponent)
        
    player=boardState.CurrentPlayer
    finishedInOne=0
    outMoveID=-1
    while(boardState.Finished==0):
        if(noOutput==False):
            if col:print("\n"+idxString+": \033[1;31;48m###### Ply ",boardState.PlyNumber," (player:",str(boardState.CurrentPlayer)+") ######\033[1;37;48m")
            else:  print("\n"+idxString+": ###### Ply "             ,boardState.PlyNumber," (player:",str(boardState.CurrentPlayer)+") ######")
        boardState.Finished,moveID,posEval=functions.nextMove(boardState,boardInput,mD,Sizes,Weights,Biases,col,debug,noOutput)
        if(boardState.PlyNumber==initialPlyNumber):
            print("moveID=",moveID)
            finishedInOne=boardState.Finished
            outMoveID=moveID
        if(boardState.PlyNumber>=searchDepth):
            winner=boardState.setWinner(0)
            if(noOutput==False): print(idxString+": INFO: Anticipated search depth=",searchDepth," reached. Get Evaluation")
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
    return outMoveID,won,finishedInOne

def playGame(startPly,Sizes,Weights,Biases,boardposition,mD,deb,noOut,mergeInfo,c):
    #N=len(boardpositions)-1
    exitState=0
    inputPos=boardpositions[0].getInput()
    inputID =""
    winners =np.zeros(1882,dtype=np.float64)
    counters=np.zeros(1882,dtype=np.int64)
    currentGameID=boardpositions[0].getInputID(False)
    if startPly<checkForDoubles or len(currentGameID)<25:
        inputID=currentGameID
    #for idx in range(N):
    #    idxString=str(idx)
    #    boardposition=boardpositions[idx]
    moveID,gameWinner,finishedInOne=play(idxString,inputPos,Sizes,Weights,Biases,boardposition,mD,deb,noOut,c)
    if mergeInfo: print(idxString+": Got game",idx,": moveID,winner=",moveID,gameWinner)
    winners[moveID],counters[moveID]=mergeStuff(idx,moveID,currentGameID,gameWinner,1,winners[moveID],counters[moveID],mergeInfo)
    if finishedInOne==1:
        exitState=1
        print(idxString+": Games finished => exitState=",exitState)
        #break
    return inputPos,inputID,currentGameID,winners,counters,exitState

def runSelfPlay(boardposition,mD,it,jobNum,startPly,nGames,Sizes,Weights,Biases,deb,noOut,mergeInfo,c):
    print("\nPlaying from next position. startPly=",startPly)
    bestPosFile="data/bestLastMoves_"+jobNum+".pkl"
    if(startPly>0):
        pieceBoardPositions = loadBestPos(bestPosFile)
        #for i,bp in enumerate(boardpositions):
        #    bp.reconstructGameState(i,pieceBoardPositions[0],startPly,gameMode)
        #    bp.nextPly()
        boardposition.reconstructGameState(pieceBoardPositions[0],startPly,gameMode)
        boardposition.nextPly()
    else:
        #for i,bp in enumerate(boardpositions):
        #    bp.initializeBoard(i,gameMode,c,deb,noOut)
        boardposition.initializeBoard(gameMode,c,deb,noOut)
    inputPos,inputID,currentGameID,gameWinners,gameCounters,exitState=playGame(startPly,Sizes,Weights,Biases,boardposition,mD,deb,noOut,mergeInfo,c)
    print("\nAll games starting at ply",startPly,"(jobNum="+jobNum+") are finished! Write the data to file.\n")
    
    #Update gameDataFile
    if os.path.exists(gameDataFile):
        print("INFO:",gameDataFile,"already exists. Import!")
        fileIn = open(gameDataFile,"rb")
        merged_data = pickle.load(fileIn)
        fileIn.close()
        if inputID!="":
            print("Merging position")
            merged_data=update_data(0,merged_data,inputPos,gameWinners,inputID,gameCounters,False)
        else:
            print("Dont store this ID.")
            merged_data[0].append(inputPos)
            merged_data[1].append(gameWinners)
    else:
        print("INFO: No",gameDataFile,"yet. Nothing to import!")
        merged_data=[[inputPos],[gameWinners],[inputID],[gameCounters]]
    fileOut = open(gameDataFile, "wb")
    pickle.dump(merged_data,fileOut)
    fileOut.close()
          
    #Write the most promising position to file
    posToWrite=[]
    idWinPerc=[(moveID,round(num,4)) for moveID,num in enumerate(gameWinners)]
    #for idWin in idWinPerc:
    #    print("moveID (win percentage)=",idWin[0],": ("+str(idWin[1])+")")
    for i in range(1):
        maxIndex=np.argmax(gameWinners)
        maxVal=gameWinners[maxIndex]
        ### need to play best move and get nnInput position
        boardPos=boardpositions[-1]
        chessBoard=boardPos.ChessBoard
        currentPlayer=boardPos.CurrentPlayer
        currentOpp   =boardPos.CurrentOpponent
        # reconstruct move now!
        dictMove=mL[maxIndex]
        capturedPiece=chessBoard[dictMove[4]][dictMove[3]]*currentOpp
        pToMove      =chessBoard[dictMove[2]][dictMove[1]]*currentPlayer
        if dictMove==(1,4,0,2,0):             #castling short
            move = nb.typed.List([1,1, 4,0, 2,0, 0, 1,0, 0, maxIndex])
        elif dictMove==(1,4,0,6,0):           #castling long
            move = nb.typed.List([1,1, 4,0, 6,0, 0, 0,1, 0, maxIndex])
        elif dictMove[0]==6:                  #enpassant move
            move = nb.typed.List([6,      6,           dictMove[1],dictMove[2], dictMove[3],dictMove[4], 6,             0,0, 1, maxIndex])
        elif dictMove[0]>1 and dictMove[0]<6: #pawn promotion
            move = nb.typed.List([6,      dictMove[0], dictMove[1],dictMove[2], dictMove[3],dictMove[4], capturedPiece, 0,0, 0, maxIndex])
        else:
            move = nb.typed.List([pToMove,pToMove,     dictMove[1],dictMove[2], dictMove[3],dictMove[4], capturedPiece, 0,0, 0, maxIndex])
        if mergeInfo: print("Reconstructed best Move:",move,"with win probability",maxVal,". Play the move now")
        boardPos.playMove(move, False, deb, noOut)
        maxPosition=boardPos.getInput()
        ID=boardPos.getInputID(deb)
        if mergeInfo: print("Writing ID=",ID,"with maxVal=",maxVal,"to file.")
        posToWrite.append(maxPosition)
    writeBestPosToFile(posToWrite,bestPosFile)
    print("INFO: len(inputPositions/Winners/IDs/Counters)=",len(merged_data[0]),"/",len(merged_data[1]),"/",len(merged_data[2]),"/",len(merged_data[3]))
    return exitState
        
### Start the code        
exitState=0
print("Iteration=",iteration)
for startPly in range(upToPly):
    if exitState==0:
        start_timeit = timer()
        exitState=runSelfPlay(boardposition,mD,iteration,jobNum,startPly,nGames,Sizes,Weights,Biases,Debug,NoOutput,PrintMergeInfo,Col)
        print("Time for",nGames,"games (startPly=",startPly,"):",timer()-start_timeit)
    else:
        print("exitState =",exitState,"! Breaking loop!")
        break
