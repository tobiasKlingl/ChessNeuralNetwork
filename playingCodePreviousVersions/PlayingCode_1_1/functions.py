import numpy as np
import random
import copy
import time
#from colorama import init, Fore, Back, Style

def getPlayerColor(player):
    if  (player==0): color="W"
    elif(player==1): color="B"
    else:print("(ERROR (functions (getPlayerColor)): invalid argument for player=",player)
    return color

def getOpponent(player):
    if  (player==0):opponent=1
    elif(player==1):opponent=0
    else:print("(ERROR (function (getOpponent): invalid argument for player=",player)
    return opponent

def nextPly(plyNumber,player):
    plyNumber+=1
    player=getOpponent(player)
    return plyNumber,player

def getReadablePosition(player,move,debug=False):
    if  (move[0]==0): xpos="A"
    elif(move[0]==1): xpos="B"
    elif(move[0]==2): xpos="C"
    elif(move[0]==3): xpos="D"
    elif(move[0]==4): xpos="E"
    elif(move[0]==5): xpos="F"
    elif(move[0]==6): xpos="G"
    elif(move[0]==7): xpos="H"
    else:print("ERROR (functions (getReadablePosition)): move[0]=",move[0]," is unknown!")
    if  (player==0): ypos=str(move[1]+1)
    elif(player==1): ypos=str(7-move[1]+1)
    readablePosition=xpos+ypos
    if(debug): print("DEBUG (functions (getReadablePosition)): Converted position:",move,"into:",readablePosition)
    return readablePosition

def getPieceUnicode(pieceAbb,colored):
    pieceUnicode=" "
    if(colored==True):
        if  ("k" in pieceAbb): pieceUnicode='\u265A'
        elif("q" in pieceAbb): pieceUnicode='\u265B'
        elif("r" in pieceAbb): pieceUnicode='\u265C'
        elif("b" in pieceAbb): pieceUnicode='\u265D'
        elif("n" in pieceAbb): pieceUnicode='\u265E'
        elif("p" in pieceAbb): pieceUnicode='\u265F'
    else:
        if("W" in pieceAbb): x=6
        elif("B" in pieceAbb): x=0
        if  ("k" in pieceAbb): pieceUnicode="".join(chr(9812+0+x))
        elif("q" in pieceAbb): pieceUnicode="".join(chr(9812+1+x))
        elif("r" in pieceAbb): pieceUnicode="".join(chr(9812+2+x))
        elif("b" in pieceAbb): pieceUnicode="".join(chr(9812+3+x))
        elif("n" in pieceAbb): pieceUnicode="".join(chr(9812+4+x))
        elif("p" in pieceAbb): pieceUnicode="".join(chr(9812+5+x))
    return pieceUnicode

def createPieceListforPrintOut(readableMoveList,prob,colored=False,debug=False):
    sortedList=[]
    pawnMoves  ="  PAWN: "
    knightMoves="  KNIGHT: "
    bishopMoves="  BISHOP: "
    rookMoves  ="  ROOK: "
    queenMoves ="  QUEEN: "
    kingMoves  ="  KING: "
    for ID,move in enumerate(readableMoveList):
        if colored:
            moveWithID="\033[1;34;49m"+move[1]+"->"+move[2]+"\033[1;37;49m(\033[1;32;49m"+str(ID)+",\033[1;37;49m"+str(prob)+") ; "
        else:
            moveWithID=move[1]+"->"+move[2]+"("+str(ID)+","+str(prob)+") ; "
        if  ("pawn"  in move): pawnMoves+=moveWithID
        elif("knight"in move): knightMoves+=moveWithID
        elif("bishop"in move): bishopMoves+=moveWithID
        elif("rook"  in move): rookMoves+=moveWithID
        elif("queen" in move): queenMoves+=moveWithID
        elif("king"  in move): kingMoves+=moveWithID
        else:
            print("ERROR (functions (createPieceListforPrintOut)): unknown pieceName")
    sortedList.append(pawnMoves)
    sortedList.append(knightMoves)
    sortedList.append(bishopMoves)
    sortedList.append(rookMoves)
    sortedList.append(queenMoves)
    sortedList.append(kingMoves)
    return sortedList

def printMoves(player,moveList,colored,probNormed,noOutputMode=False,debug=False):
    readableMoveList=[]
    if(debug): print("moveList=",moveList)
    for i,move in enumerate(moveList):
        piece=move[0]
        pieceNew=move[1]
        capturedMove=move[4]
        castling=move[5]
        enPassant=move[6]
        prob=probNormed[i]
        pos_before=getReadablePosition(player,move[2],debug)
        pos_after=getReadablePosition(player,move[3],debug)
        #if(piece=="king" and pos_before=="E1" and (pos_after=="G1" or pos_after=="C1")):
        #elif(piece=="king" and pos_before=="E8" and (pos_after=="G8" or pos_after=="C8")):
        if(castling[0]==True):
            readableMove=[piece,pos_before,pos_after+"(White CASTLING long)"]
        elif(castling[1]==True):
            readableMove=[piece,pos_before,pos_after+"(Black CASTLING short)"]
        else:
            readableMove=[piece,pos_before,pos_after]
        if(capturedMove==True):
            if(enPassant==True):
                readableMove[2]+="(Piece CAPTURED via en-passant)"
            elif(piece=="pawn" and pieceNew!="pawn"):
                readableMove[2]+="(Piece CAPTURED + promtion: "+pieceNew+")"
            else:
                readableMove[2]+="(Piece CAPTURED)"
        else:
            if(piece=="pawn" and pieceNew!="pawn"):
                readableMove[2]+="(promotion: "+pieceNew+")"
        if(debug): print("DEBUG (functions (printMoves)): readableMove=",readableMove)
        readableMoveList.append(readableMove)
    if(noOutputMode==False):
        sortedLists=createPieceListforPrintOut(readableMoveList,prob,colored,debug)
        print("INFO (functions (printMoves)): Moves for player",player,":")
        for list in sortedLists:
            print(list)

def moveProb(player, boardpositions, move, debug, noOutputMode):
    if(debug): print("(functions (inCheck)): move=",move)
    copiedBoardPositions=copy.deepcopy(boardpositions)
    #moveInfo=copiedBoardPositions.playMove(player, move, [False,False], isEnPassant, False, debug)
    moveInfo=copiedBoardPositions.playMove(player, move, False, debug, noOutputMode)
    willPlayerBeInCheck=copiedBoardPositions.isPlayerInCheck(player,False)
    if willPlayerBeInCheck:
        moveProbability=-1.0
    else:
        if("SelfplayNetwork" in copiedBoardPositions.GameMode and player==0):
            if(noOutputMode==False): print("Neural network is moving now!")
            ### Replace this part in the future...
            copiedBoardPositions.reverseBoard()
            opponent=getOpponent(player)
            inputForNN=np.reshape(copiedBoardPositions.getInput(opponent,debug,noOutputMode), (780, 1))
            ### with the following line: 
            #inputForNN=np.reshape(copiedBoardPositions.getInput(player,debug,noOutputMode), (780, 1))
            moveProbability=copiedBoardPositions.NetWork.feedforward(inputForNN)[0][0]
            print("Probability to win with this move is:",moveProbability)
        else:
            moveProbability=1.0
    if(debug): print("(functions (inCheck)): willPlayerBeInCheck?",willPlayerBeInCheck,"(moveProbability=",moveProbability,"). If so remove move from list.")
    return willPlayerBeInCheck,moveProbability

def getValidMoves(player,boardpositions,moveCandidates,castlingCandidates,enPassantCandidates,debug=False,noOutputMode=False):
    moveID=0
    castlingLongID=-999
    castlingShortID=-999
    enPassantIDs=[-999,-999]
    outOfCheckMoves=[]
    for testMove in moveCandidates: # test normal moves
        if(debug and noOutputMode==False): print("(functions (nextMove)): testMove=",testMove)
        willPlayerBeInCheck,prob=moveProb(player, boardpositions, testMove, debug, noOutputMode)
        testMove.append(prob)
        if(willPlayerBeInCheck==False):
            outOfCheckMoves.append(testMove)
            moveID+=1
    for castlingMove in castlingCandidates: # test castling moves
        if(debug and noOutputMode==False): print("DEBUG (functions (nextMove)): castlingMove=",castlingMove)
        startMove        =[castlingMove[0],castlingMove[1],castlingMove[2],[4,0],castlingMove[4],castlingMove[5],castlingMove[6]]
        if(castlingMove[3]==[2,0]): #castling long
            inBetweenMove=[castlingMove[0],castlingMove[1],castlingMove[2],[3,0],castlingMove[4],castlingMove[5],castlingMove[6]]
        elif(castlingMove[3]==[6,0]): #castling short
            inBetweenMove=[castlingMove[0],castlingMove[1],castlingMove[2],[5,0],castlingMove[4],castlingMove[5],castlingMove[6]]
        willPlayerBeInCheck1,prob1=moveProb(player, boardpositions, startMove    , debug, noOutputMode)
        willPlayerBeInCheck2,prob2=moveProb(player, boardpositions, inBetweenMove, debug, noOutputMode)
        willPlayerBeInCheck3,prob3=moveProb(player, boardpositions, castlingMove , debug, noOutputMode)
        if(willPlayerBeInCheck1==False and willPlayerBeInCheck2==False and willPlayerBeInCheck3==False):
            if(castlingMove[3]==[2,0]):
                castlingLongID=moveID
            elif(castlingMove[3]==[6,0]):
                castlingShortID=moveID
            moveID+=1
            castlingMove.append(prob3)
            outOfCheckMoves.append(castlingMove)
    if(debug): print("INFO (functions (nextMove)): len(enPassantCandidates)=",len(enPassantCandidates))
    for num,enPassantMove in enumerate(enPassantCandidates): # test enPassant moves
        if(debug and noOutputMode==False): print("DEBUG (functions (nextMove)): enPassantMove=",enPassantMove)
        willPlayerBeInCheck,prob=moveProb(player, boardpositions, enPassantMove, debug, noOutputMode)
        if(willPlayerBeInCheck==False):
            enPassantIDs[num]=moveID
            moveID+=1
            enPassantMove.append(prob)
            outOfCheckMoves.append(enPassantMove)
    return outOfCheckMoves,castlingLongID,castlingShortID,enPassantIDs

def nextMove(player,boardpositions,colored=False,debug=False,noOutputMode=False):
    gameMode=boardpositions.GameMode
    finished=False
    playerInCheck=boardpositions.isPlayerInCheck(player,debug)
    if(playerInCheck==True and noOutputMode==False): print("Player",player,"is in CHECK!")
    ### GET VALID MOVES ###
    moveCandidates,castlingCandidates,enPassantCandidates=boardpositions.findAllowedMoves(player, debug)
    outOfCheckMoves,castlingLongID,castlingShortID,enPassantIDs=getValidMoves(player,boardpositions,moveCandidates,castlingCandidates,enPassantCandidates,debug,noOutputMode)
    ### PLAY THE MOVE ###
    if(len(outOfCheckMoves)>0):
        maxNum=len(outOfCheckMoves)
        prob=[move[7] for move in outOfCheckMoves]
        s = sum(prob)
        probNormed = [float(i)/s for i in prob]
        prob_rounded = [round(num, 3) for num in prob]
        probNormed_rounded = [round(num, 3) for num in probNormed]
        if(noOutputMode==False): print("prob =",prob_rounded)
        if(noOutputMode==False): print("normd=",probNormed_rounded)
        if(gameMode=="WhiteVsBlack" or (gameMode=="WhiteVsComputer" and player==0) or (gameMode=="BlackVsComputer" and player==1)):
            while True:
                try:
                    rand= int(input("Input your move:"))
                    if rand>=maxNum:
                        print("Your integer is too large. Choose between 0 and",maxNum-1)
                        continue
                    else:
                        break
                except ValueError:
                    print("Not an integer!")  
                    continue
            print("Choose move ID",rand)
            move=outOfCheckMoves[rand]
        else:
            if(  gameMode=="SelfplayRandomVsRandom"   or (gameMode=="SelfplayNetworkVsRandom" and player==1)):
                #print("Random move!")
                rand=random.randint(0,maxNum-1)
                move=outOfCheckMoves[rand]
            elif(gameMode=="SelfplayNetworkVsNetwork" or (gameMode=="SelfplayNetworkVsRandom" and player==0)):
                print("len(probNormed)=",len(probNormed))
                print("maxNum=",maxNum)
                np.random.choice(np.arange(maxNum), p=probNormed)
                rand=random.randint(0,maxNum-1)
                move=outOfCheckMoves[rand]
            else:
                time.sleep(1.5)
                rand=random.randint(0,maxNum)
                move=outOfCheckMoves[rand]
            if(debug): print("(functions (nextMove)): rand=",rand)
            if(noOutputMode==False): print("INFO (functions (nextMove): IDs: rand,castLong,castShort,enpassant=",rand,castlingLongID,castlingShortID,enPassantIDs)
        printMoves(player,outOfCheckMoves,colored,probNormed_rounded,noOutputMode,debug)
        moveInfoList=boardpositions.playMove(player, move, True, debug, noOutputMode)
        if(noOutputMode==False):
            boardpositions.printChessBoard(player,colored,moveInfoList,debug)
    elif(len(outOfCheckMoves)==0 and playerInCheck==True):
        if(noOutputMode==False): print("Player",player,"is CHECKMATE!")
        boardpositions.setWinner(getOpponent(player))
        finished=True
    elif(len(outOfCheckMoves)==0 and playerInCheck==False):
        if(noOutputMode==False): print("Player",player,"has no more moves available => Remis.")
        boardpositions.setWinner(-1)
        finished=True

    return finished
