import numpy as np
import BoardPositions as bp
import functions
#import time
from timeit import default_timer as timer
import pickle
import os
import random
import sys
sys.path.append("../NetworkCode/")
#import NNlib_1_4 as nn


debug=False
noOutputMode=False
colored=True

gameMode="SelfplayRandomVsRandom"
#gameMode="SelfplayNetworkVsRandom"
#gameMode="VsComputer"
#gameMode="WhiteVsBlack"

argin=sys.argv
try:
    numberOfGamesToPlay=int(argin[1])
    StopGameNumber=int(argin[2])
    checkForDouble=int(argin[3])
    gameMode=argin[4]
except:
    numberOfGamesToPlay=50
    StopGameNumber=200
    checkForDouble=5
    pass  

def updateOutput(stuff, allPos, allIDs, allOutput, allCounter):
    pos    =stuff[0]
    IDs    =stuff[1]
    output =stuff[2]
    counter=stuff[3]
    for i,ID in enumerate(IDs):
        alreadyIncluded=False
        for j,IDAll in enumerate(allIDs):
            if ID==IDAll:
                print("INFO: This position (",ID,") has been played",allCounter[j],"times before (position"+str(j)+").")
                print("output[" +str(i)+"]=",output[i] ,", allOutput[" +str(j)+"]=",allOutput[j])
                print("counter["+str(i)+"]=",counter[i],", allCounter["+str(j)+"]=",allCounter[j])
                allOutput[j]=(output[i]*counter[i]+allOutput[j]*allCounter[j])/(counter[i]+allCounter[j])
                allCounter[j]+=counter[i]
                print("new allOutput["+str(j)+"]=",allOutput[j],", allCounter[",j,"]=",allCounter[j])
                alreadyIncluded=True
                break
        if alreadyIncluded==False:
            print("INFO: Adding new position (",ID,") at position:",len(allIDs),". i,output[",i,"],counter[",i,"]=",i,output[i],counter[i])
            allPos.insert(    len(allIDs),pos[i])
            allOutput.insert( len(allIDs),output[i])
            allCounter.insert(len(allIDs),counter[i])
            allIDs.insert(    len(allIDs),IDs[i])
    allPos+=pos[len(IDs):]
    allOutput+=output[len(IDs):]
    return allPos,allIDs,allOutput,allCounter

def getGamePositions(winner,gamePositions,gamePositionIDs):
    trainingPositions  ,trainingPositionIDs  ,trainingOutput  ,trainingCounter  =[],[],[],[]
    validationPositions,validationPositionIDs,validationOutput,validationCounter=[],[],[],[]
    testPositions      ,testPositionIDs      ,testOutput      ,testCounter      =[],[],[],[]
    lenIDs=len(gamePositionIDs)
    lenPos=len(gamePositions)
    for i in range(lenPos):
        if(winner==0.5):
            out=0.5
        elif((winner==0 and i%2==0) or (winner==1 and i%2==1)):
            out=1.0
        elif((winner==0 and i%2==1) or (winner==1 and i%2==0)):
            out=0.0
        else:
            out=-1.0
            print("ERROR: Not a valid output combination#! for out=",out)
        randm=random.randint(1,100)
        if(i==0): print("INFO: gamePositionIDs=",gamePositionIDs)
        if i<lenIDs: print("INFO: i,gamePositionIDs[",i,"]=",i,gamePositionIDs[i])
        if(randm>=10):
            trainingPositions.append(gamePositions[i])
            trainingOutput.append(out)
            if i<lenIDs:
                print("INFO: Adding to Training")
                trainingPositionIDs.append(gamePositionIDs[i])
                trainingCounter.append(1)
        elif(randm>=0): # Increase to value >0 to also add to test data
            validationPositions.append(gamePositions[i])
            validationOutput.append(out)
            if i<lenIDs:
                print("INFO: Adding to Validation")
                validationPositionIDs.append(gamePositionIDs[i])
                validationCounter.append(1)
        else:
            testPositions.append(gamePositions[i])
            testOutput.append(out)
            if i<lenIDs:
                print("INFO: Adding to Test")
                testPositionIDs.append(gamePositionIDs[i])
                testCounter.append(1)
    trainStuff=[trainingPositions  ,trainingPositionIDs  ,trainingOutput  ,trainingCounter]
    validStuff=[validationPositions,validationPositionIDs,validationOutput,validationCounter]
    testStuff =[testPositions      ,testPositionIDs      ,testOutput      ,testCounter]
    return trainStuff,validStuff,testStuff
                
def play(gameMode):
    castling=[[True,True],[True,True]] #Castling for [[white long,white short],[black long, black short]] still allowed?
    enPassant=[False,[-1,-1]]          #Enpassant allowed in next move; Enpassant[1] is given from opponents view.
    finished=False                     #Set to True when the game is over
    player=0                           #Player to move: 0=white, 1=black
    plyNumber=1
    
    boardpositions=bp.BoardPositions(8,8,6)
    boardpositions.definePieceBoards(debug,noOutputMode)
    boardpositions.initializeBoard(colored,debug,noOutputMode)
    gamePositions,gamePositionIDs=[],[]
    while(finished==False):
        if("Selfplay" in gameMode):
            if plyNumber>1: #Disregard first position!
                gamePositions.append(boardpositions.getInput(player,enPassant,castling,debug,noOutputMode))
                if plyNumber<=checkForDouble:
                    gamePositionIDs.append(boardpositions.getInputID(player,enPassant,castling,debug,noOutputMode))
        if(noOutputMode==False):
            if colored:
                print("\n\033[1;32;48m###### Ply ",plyNumber," (player:",player,") ######\033[1;37;48m")
            else:
                print("\n###### Ply ",plyNumber," (player:",player,") ######")
        castling,enPassant,finished=functions.nextMove(gameMode,player,castling,enPassant,boardpositions,colored,debug,noOutputMode)
        if(plyNumber>=StopGameNumber):
            winner=boardpositions.setWinner(0.5)
            if(noOutputMode==False):
                print("INFO: More than",StopGameNumber,"plys played. This is a draw.")
            break
        else:
            plyNumber,player=functions.nextPly(plyNumber,player)
            boardpositions.reverseBoard()
    winner=boardpositions.getWinner()
    textColor,resetColor="",""
    if(colored==True): textColor,resetColor="\033[1;31;49m","\033[1;37;49m"
    if(winner==0):  print(textColor,"White won this game!!! in",plyNumber,"plys.",resetColor)
    elif(winner==1):print(textColor,"Black won this game!!! in",plyNumber,"plys.",resetColor)
    else:           print(textColor,"Game ended remis!!!",resetColor)
    if((winner==0 or winner==1) and "Selfplay" in gameMode):
        trainStuff,validStuff,testStuff=getGamePositions(winner,gamePositions,gamePositionIDs)
    else:
        trainStuff,validStuff,testStuff=[],[],[]
    return winner,trainStuff,validStuff,testStuff


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
    if os.path.exists("data/training_data.pkl"):
        print("INFO: data/training_data.pkl already exists. Import!")
        fileIn = open("data/training_data.pkl","rb")
        allTrainStuff,allValidStuff,allTestStuff = pickle.load(fileIn)
        allTrainPos,allTrainOut,allTrainIDs,allTrainCounter=allTrainStuff[0],allTrainStuff[1],allTrainStuff[2],allTrainStuff[3]
        allValidPos,allValidOut,allValidIDs,allValidCounter=allValidStuff[0],allValidStuff[1],allValidStuff[2],allValidStuff[3]
        allTestPos ,allTestOut ,allTestIDs ,allTestCounter =allTestStuff[0] ,allTestStuff[1] ,allTestStuff[2] ,allTestStuff[3]
        fileIn.close()
    else:
        print("INFO: No data/training_data.pkl yet. Nothing to import!")
        allTrainPos,allTrainIDs,allTrainOut,allTrainCounter=[],[],[],[]
        allValidPos,allValidIDs,allValidOut,allValidCounter=[],[],[],[]
        allTestPos ,allTestIDs ,allTestOut ,allTestCounter =[],[],[],[]
    print("INFO: len(allTrainPos/IDs/Out/Counter)=",len(allTrainPos),"/",len(allTrainIDs),"/",len(allTrainOut),"/",len(allTrainCounter))
    print("INFO: len(allValidPos/IDs/Out/Counter)=",len(allValidPos),"/",len(allValidIDs),"/",len(allValidOut),"/",len(allValidCounter))
    print("INFO: len(allTestPos/IDs/Out/Counter)=" ,len(allTestPos) ,"/",len(allTestIDs) ,"/",len(allTestOut) ,"/",len(allTestCounter))


    #Inputname = "savedir+subdir+NN_SMGD_L2_maxEpochs_10_nEpochs_1_numEtaRed_7_miniBatchSize_10_etaInit_0.5_lmbda_5.0_mu_0.2"
    #net = nn.load("Inputname")
    #self.feedforward(x)

    for i in range(numberOfGamesToPlay):
        print("### Playing game number",i,"###")
        start_timeit = timer()
        winner,trainStuff,validStuff,testStuff=play(gameMode)
        if winner==0 or winner==1:
            print("INFO: Update TRAINING output!")
            allTrainPos,allTrainIDs,allTrainOut,allTrainCounter=updateOutput(trainStuff, allTrainPos, allTrainIDs, allTrainOut, allTrainCounter)
            print("INFO: Update VALIDATION output!")
            allValidPos,allValidIDs,allValidOut,allValidCounter=updateOutput(validStuff, allValidPos, allValidIDs, allValidOut, allValidCounter)
            print("INFO: Update TEST output!")
            allTestPos ,allTestIDs ,allTestOut ,allTestCounter =updateOutput(testStuff , allTestPos , allTestIDs , allTestOut , allTestCounter)
            print("INFO: len(allTrainPos/IDs/Out/Counter)=",len(allTrainPos),"/",len(allTrainIDs),"/",len(allTrainOut),"/",len(allTrainCounter))
            print("INFO: len(allValidPos/IDs/Out/Counter)=",len(allValidPos),"/",len(allValidIDs),"/",len(allValidOut),"/",len(allValidCounter))
            print("INFO: len(allTestPos/IDs/Out/Counter)=" ,len(allTestPos) ,"/",len(allTestIDs) ,"/",len(allTestOut) ,"/",len(allTestCounter))
        print("Time for for game=",timer()-start_timeit)

    if(allTrainPos or allValidPos or allTestPos):
        allTrainings_data =[allTrainPos,allTrainOut,allTrainIDs,allTrainCounter]
        allValidation_data=[allValidPos,allValidOut,allValidIDs,allValidCounter]
        allTest_data      =[allTestPos ,allTestOut ,allTestIDs ,allTestCounter]
        fileOut = open("data/training_data_new.pkl", "wb")
        pickle.dump([allTrainings_data,allValidation_data,allTest_data],fileOut)
        fileOut.close()
    else:
        print("No games won be either side! No output.")
else:
    play(gameMode)
