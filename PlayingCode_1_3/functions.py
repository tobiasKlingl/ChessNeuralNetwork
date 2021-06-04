import numpy as np
import random
from timeit import default_timer as timer

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
            moveWithID="\033[1;34;49m"+move[1]+"->"+move[2]+"\033[1;37;49m(\033[1;32;49m"+str(ID)+",\033[1;37;49m"+str(prob[ID])+") ; "
        else:
            moveWithID=move[1]+"->"+move[2]+"("+str(ID)+","+str(prob[ID])+") ; "
        if  ("p" in move): pawnMoves+=moveWithID
        elif("n" in move): knightMoves+=moveWithID
        elif("b" in move): bishopMoves+=moveWithID
        elif("r" in move): rookMoves+=moveWithID
        elif("q" in move): queenMoves+=moveWithID
        elif("k" in move): kingMoves+=moveWithID
        else:
            print("ERROR (functions (createPieceListforPrintOut)): unknown pieceName")
    sortedList.append(pawnMoves)
    sortedList.append(knightMoves)
    sortedList.append(bishopMoves)
    sortedList.append(rookMoves)
    sortedList.append(queenMoves)
    sortedList.append(kingMoves)
    return sortedList

def printMoves(player,moveList,colored,probNormed,debug=False):
    readableMoveList=[]
    if(debug): print("moveList=",moveList)
    for i,move in enumerate(moveList):
        piece=move[0]
        pieceNew=move[1]
        capturedPiece=move[4]
        castling=move[5]
        enPassant=move[6]
        pos_before=getReadablePosition(player,move[2],debug)
        pos_after=getReadablePosition(player,move[3],debug)
        if(castling[0]==True):
            readableMove=[piece,pos_before,pos_after+"(White CASTLING long)"]
        elif(castling[1]==True):
            readableMove=[piece,pos_before,pos_after+"(Black CASTLING short)"]
        else:
            readableMove=[piece,pos_before,pos_after]
        if(capturedPiece!=""):
            if(enPassant==True):
                readableMove[2]+="(Piece CAPTURED via en-passant)"
            elif(piece=="p" and pieceNew!="p"):
                readableMove[2]+="(Piece CAPTURED + promtion: "+pieceNew+")"
            else:
                readableMove[2]+="(Piece CAPTURED)"
        else:
            if(piece=="p" and pieceNew!="p"):
                readableMove[2]+="(promotion: "+pieceNew+")"
        if(debug): print("DEBUG (functions (printMoves)): readableMove=",readableMove)
        readableMoveList.append(readableMove)
    sortedLists=createPieceListforPrintOut(readableMoveList,probNormed,colored,debug)
    print("INFO (functions (printMoves)): Moves for player",player,":")
    for list in sortedLists:
        print(list)

def getNNInput(boardpositions, move, debug, noOutputMode):
    isPlayerInCheckBeforeMove=boardpositions.IsPlayerInCheck
    castlingBeforeMove = [x[:] for x in boardpositions.Castling]
    enPassantBeforeMove = [boardpositions.EnPassant[0], boardpositions.EnPassant[1][:]]
   
    moveInfo,captPiece=boardpositions.playMove(move, False, debug, noOutputMode)
    boardpositions.setIsPlayerInCheck()
    willPlayerBeInCheck=boardpositions.IsPlayerInCheck

    if("SelfplayNetwork" in boardpositions.GameMode):
         nnInput=boardpositions.getInput() #Potential performance boost in this function
    else:
        nnInput=1.0
        
    #reverseThePlayedMove
    boardpositions.reverseMove(move, captPiece, castlingBeforeMove, enPassantBeforeMove)
    boardpositions.IsPlayerInCheck=isPlayerInCheckBeforeMove
    boardpositions.Castling=castlingBeforeMove
    boardpositions.EnPassant=enPassantBeforeMove
    return willPlayerBeInCheck,nnInput

def moveProbs(boardpositions,net,nnInputs,debug):
    if("SelfplayNetwork" in boardpositions.GameMode and boardpositions.CurrentPlayer==0):
        probs=net.feedforward(nnInputs)[0]
        moveProbabilities=probs.tolist()
        if(debug): print("Probabilities to win:",moveProbabilities)
    else:
        moveProbabilities=[1.0 for i in nnInputs]
    return moveProbabilities

def nextMove(net,boardpositions,colored=False,debug=False,noOutputMode=False):
    #moveStartTime=timer()
    gameMode=boardpositions.GameMode
    player=boardpositions.CurrentPlayer
    finished=False
    boardpositions.setIsPlayerInCheck(debug)
    playerInCheck=boardpositions.IsPlayerInCheck
    if(playerInCheck==True and noOutputMode==False): print("Player",player,"is in CHECK!")
    outOfCheckMoves,moveProbabilities=boardpositions.findAllowedMoves(net,debug,noOutputMode)
    if(len(outOfCheckMoves)>0):
        if(len(outOfCheckMoves)<=8):
            #Check if only king is left and opponent can still win!
            if(noOutputMode==False): print("Less than 9 moves available! Check if game is decided.")
            colorPlayer  =boardpositions.PlayerColors[boardpositions.CurrentPlayer]
            colorOpponent=boardpositions.PlayerColors[boardpositions.CurrentOpponent]
            Knight,Bishop,Rook,Queen,Pawn=colorOpponent+"n",colorOpponent+"b",colorOpponent+"r",colorOpponent+"q",colorOpponent+"p"
            playerPieces,opponentPieces=[],[]
            for row in boardpositions.ChessBoard:
                for col in row:
                    if colorPlayer in col:
                        playerPieces.append(col)
                    if colorOpponent in col:
                        opponentPieces.append(col)
            if(noOutputMode==False): print("playerPieces=",playerPieces)
            if(len(playerPieces)==1):
                if(noOutputMode==False): print("opponentPieces=",opponentPieces)
                if(len(opponentPieces)>2 or
                   len(opponentPieces)>1 and (Rook in opponentPieces or Queen in opponentPieces)):# or Pawn in opponentPieces)):
                    for rand,move in enumerate(outOfCheckMoves):
                        if move[4]!="":
                            moveInfoList,captPiece=boardpositions.playMove(move, True, debug, noOutputMode)
                            if(noOutputMode==False): boardpositions.printChessBoard(rand,-1.0,colored,moveInfoList,debug)
                            return False
                    boardpositions.setWinner(boardpositions.CurrentOpponent)
                    return True
                elif(len(opponentPieces)==2 and (Bishop in opponentPieces or Knight in opponentPieces)):
                    boardpositions.setWinner(0.5)
                    return True
        maxNum=len(outOfCheckMoves)
        s = sum(moveProbabilities)
        probNormed = [float(i)/s for i in moveProbabilities]
        prob_rounded = [round(num, 3) for num in moveProbabilities]
        #probNormed = moveProbabilities/np.sum(moveProbabilities)
        #prob_rounded = np.round(moveProbabilities,3)
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
            if(  gameMode=="SelfplayRandomVsRandom" or (gameMode=="SelfplayNetworkVsRandom" and player==1)):
                if(noOutputMode==False): print("Random mover is moving now!")
                rand=random.randint(0,maxNum-1)
                move=outOfCheckMoves[rand]
            elif(gameMode=="SelfplayNetworkVsNetwork" or (gameMode=="SelfplayNetworkVsRandom" and player==0)):
                if(noOutputMode==False): print("Neural network is moving now!")
                #np.random.choice(np.arange(maxNum), p=probNormed)
                #rand=random.randint(0,maxNum-1)
                rand=np.argmax(moveProbabilities)
                move=outOfCheckMoves[rand]
            else:
                rand=random.randint(0,maxNum)
                move=outOfCheckMoves[rand]
            if(debug): print("(functions (nextMove)): rand=",rand)
        if noOutputMode==False:printMoves(player,outOfCheckMoves,colored,prob_rounded,debug)
        moveInfoList,captPiece=boardpositions.playMove(move, True, debug, noOutputMode)
        if(noOutputMode==False):
            boardpositions.printChessBoard(rand,prob_rounded[rand],colored,moveInfoList,debug)
    elif(len(outOfCheckMoves)==0 and playerInCheck==True):
        if(noOutputMode==False): print("Player",player,"is CHECKMATE!")
        boardpositions.setWinner(boardpositions.CurrentOpponent)
        finished=True
    elif(len(outOfCheckMoves)==0 and playerInCheck==False):
        if(noOutputMode==False): print("Player",player,"has no more moves available => Remis.")
        boardpositions.setWinner(0.5)
        finished=True
    #print("time nextMove=",timer()-moveStartTime)
    return finished
