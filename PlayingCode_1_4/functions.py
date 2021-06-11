import numpy as np
import random
from timeit import default_timer as timer
from numba import jit,njit,types,int64,typed

    
@njit(cache=True)
def getPlayerNumerator(player):
    if  (player==+1): numerator=0
    elif(player==-1): numerator=1
    else:print("(ERROR (functions (getPlayerColor)): invalid argument for player=",player)
    return numerator

@njit(cache=True)
def getOpponent(player):
    if  (player==+1):opponent=-1
    elif(player==-1):opponent=+1
    else:print("(ERROR (function (getOpponent): invalid argument for player=",player)
    return opponent

@njit(cache=True)
def getReadablePosition(player,x,y,debug=False):
    if  (x==0): xpos="A"
    elif(x==1): xpos="B"
    elif(x==2): xpos="C"
    elif(x==3): xpos="D"
    elif(x==4): xpos="E"
    elif(x==5): xpos="F"
    elif(x==6): xpos="G"
    elif(x==7): xpos="H"
    else:
        xpos=""
        print("ERROR (functions (getReadablePosition)): pos[0]=",x," is unknown!")
    if  (player==+1): ypos=str(y+1)
    elif(player==-1): ypos=str(7-y+1)
    else:
        ypos=""
        print("ERROR (functions (getReadablePosition)): player=",player," is unknown!")
    readablePosition=xpos+ypos
    if(debug): print("DEBUG (functions (getReadablePosition)): Converted position:",x,y,"into:",readablePosition)
    return readablePosition

@njit(cache=True)
def getPieceName(pieceAbb):
    name=""
    pieceNumber=abs(pieceAbb)
    if   pieceNumber==1: name="king" 
    elif pieceNumber==2: name="queen" 
    elif pieceNumber==3: name="rook" 
    elif pieceNumber==4: name="bishop" 
    elif pieceNumber==5: name="night" 
    elif pieceNumber==6: name="pawn" 
    else:print("ERROR (BoardPositions (getPieceNumber)): PieceNumber",pieceNumber,"unknown!")
    return name

@njit(cache=True)
def getPieceUnicode(piece,colored):
    pieceUnicode=" "
    pieceAbb=abs(piece)
    if(colored==True):
        if  (pieceAbb==1): pieceUnicode='\u265A'
        elif(pieceAbb==2): pieceUnicode='\u265B'
        elif(pieceAbb==3): pieceUnicode='\u265C'
        elif(pieceAbb==4): pieceUnicode='\u265D'
        elif(pieceAbb==5): pieceUnicode='\u265E'
        elif(pieceAbb==6): pieceUnicode='\u265F'
    else:
        if(piece>0): x=6
        elif(piece<0): x=0
        if  (pieceAbb==1): pieceUnicode="".join(chr(9812+pieceAbb-1+x))
        elif(pieceAbb==2): pieceUnicode="".join(chr(9812+pieceAbb-1+x))
        elif(pieceAbb==3): pieceUnicode="".join(chr(9812+pieceAbb-1+x))
        elif(pieceAbb==4): pieceUnicode="".join(chr(9812+pieceAbb-1+x))
        elif(pieceAbb==5): pieceUnicode="".join(chr(9812+pieceAbb-1+x))
        elif(pieceAbb==6): pieceUnicode="".join(chr(9812+pieceAbb-1+x))
    return pieceUnicode

@njit(cache=True)
def createPieceListforPrintOut(readableMoveList,prob,colored=False,debug=False):
    SortedList=typed.List()
    pawnMoves  ="  PAWN: "
    knightMoves="  KNIGHT: "
    bishopMoves="  BISHOP: "
    rookMoves  ="  ROOK: "
    queenMoves ="  QUEEN: "
    kingMoves  ="  KING: "
    #+",\033[1;37;49m"+str(prob[ID])+") ; "
    #+","+str(prob[ID])+") ; "
    #percentages=prob.astype(np.int32)
    for ID,readableMove in enumerate(readableMoveList):
        if colored:
            moveWithID="\033[1;34;49m"+readableMove[1]+"->"+readableMove[2]+"\033[1;37;49m(\033[1;32;49m"+str(ID)+",\033[1;37;49m"+str(prob[ID])+") ; "
        else:
            moveWithID=readableMove[1]+"->"+readableMove[2]+"("+str(ID)+","+str(prob[ID])+") ; "
        if  (readableMove[0]=="pawn"):
            pawnMoves+=moveWithID
        elif(readableMove[0]=="night"):
            knightMoves+=moveWithID
        elif(readableMove[0]=="bishop"):
            bishopMoves+=moveWithID
        elif(readableMove[0]=="rook"):
            rookMoves+=moveWithID
        elif(readableMove[0]=="queen"):
            queenMoves+=moveWithID
        elif(readableMove[0]=="king"):
            kingMoves+=moveWithID
        else:
            print("ERROR (functions (createPieceListforPrintOut)): unknown pieceName")
    SortedList.append(kingMoves)
    SortedList.append(queenMoves)
    SortedList.append(rookMoves)
    SortedList.append(bishopMoves)
    SortedList.append(knightMoves)
    SortedList.append(pawnMoves)
    return SortedList

@njit(cache=True)
def printMoves(player,moveList,colored,probNormed,debug=False):
    readableMoveList=typed.List()
    if(debug): print("moveList=",moveList)
    for i,move in enumerate(moveList):
        piece=move[0]
        pieceName=getPieceName(piece)
        pieceNew=move[1]
        pieceNameNew=getPieceName(pieceNew)
        capturedPiece=move[6]
        castlingL=move[7]
        castlingS=move[8]
        enPassant=move[9]
        pos_before=getReadablePosition(player,move[2],move[3],debug)
        pos_after=getReadablePosition( player,move[4],move[5],debug)
        if(castlingL==True):
            readableMove=[pieceName,pos_before,pos_after+"(White CASTLING long)"]
        elif(castlingS==True):
            readableMove=[pieceName,pos_before,pos_after+"(Black CASTLING short)"]
        else:
            readableMove=[pieceName,pos_before,pos_after]
        if(capturedPiece!=0):
            if(enPassant!=0):
                readableMove[2]+="(Piece CAPTURED via en-passant)"
            elif(abs(piece)==6 and abs(pieceNew)!=6):
                readableMove[2]+="(Piece CAPTURED + promtion: "+pieceNameNew+")"
            else:
                readableMove[2]+="(Piece CAPTURED)"
        else:
            if(abs(piece)==6 and abs(pieceNew)!=6):
                readableMove[2]+="(promotion: "+pieceNameNew+")"
        if(debug):
            print("DEBUG (functions (printMoves)): readableMove=",readableMove)
        readableMoveList.append(readableMove)
    SortedLists=createPieceListforPrintOut(readableMoveList,probNormed,colored,debug)
    print("INFO (functions (printMoves)): Moves for player",player,":")
    for list in SortedLists:
        print(list)

@njit(cache=True)
def quickPrint(boardpositions,string):
        print("# Board after",string,"#")
        print("## A B C D E F G H #")
        for j in range(len(boardpositions.ChessBoard)):
            ChessBoardString=str(8-j)+" "
            colNum=j
            if boardpositions.CurrentPlayer==1: colNum=7-j
            printColor=""
            for i in range(8):
                pieceUnicode=getPieceUnicode(boardpositions.ChessBoard[colNum][i],False)
                ChessBoardString+=printColor+pieceUnicode+" "
            ChessBoardString+=" "+str(8-j)
            print(ChessBoardString)
        print("# A B C D E F G H ##")
        print("###### -Board ######\n")
        
@njit(cache=True)
def getNNInput(boardpositions, move, debug, noOutputMode):
    isPlayerInCheckBeforeMove=boardpositions.IsPlayerInCheck
    castlingBeforeMove = boardpositions.Castling.copy()
    enPassantBeforeMove = boardpositions.EnPassant
   
    moveInfo,captPiece=boardpositions.playMove(move, False, debug, noOutputMode)
    if(debug): quickPrint(boardpositions,"move")
    boardpositions.setIsPlayerInCheck()
    willPlayerBeInCheck=boardpositions.IsPlayerInCheck

    if("SelfplayNetwork" in boardpositions.GameMode):
        nnInput=boardpositions.getInput()
    else:
        nnInput=np.ones(780,dtype=np.int64)
    boardpositions.reverseMove(move, captPiece, castlingBeforeMove, enPassantBeforeMove, isPlayerInCheckBeforeMove)
    if(debug): quickPrint(boardpositions,"reset")
    return willPlayerBeInCheck,nnInput

@njit(cache=True)
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

@njit(cache=True)
def relu(z):
    #return np.maximum(z, 0)
    return z*(z>0)

@njit(cache=True)
def moveProbs(boardpositions,Sizes,Weights,Biases,nnInputList,debug):
    lenInput=len(nnInputList)
    if(lenInput>100):
        print("ERROR: More than 100 moves available (lenInput)",lenInput,")! Some moves are overlooked!")
    nnInput=np.zeros((100,780),dtype=np.float64)
    for i,inArr in enumerate(nnInputList):
        nnInput[i]=inArr
    Num_layers=len(Sizes)
    if("SelfplayNetwork" in boardpositions.GameMode and boardpositions.CurrentPlayer==1):
        a=np.ascontiguousarray(nnInput[:lenInput].transpose())
        for i,(b,w) in enumerate(zip(Biases, Weights)):
            #print("a=",a)
            #print("i=",i)
            if i==Num_layers-2:
                #print("sigmoid")
                a = sigmoid(w@a + b)
            else:
                #print("relu")
                a = relu(w@a + b)
        moveProbabilities=a.reshape(-1)
        if(debug):
            print("Probabilities to win:",moveProbabilities)
    else:
        probs=np.ones((lenInput,1),dtype=np.float64)
        moveProbabilities = probs.reshape(-1)
    return moveProbabilities

@njit
def nextMove(boardpositions,Sizes,Weights,Biases,colored=False,debug=False,noOutputMode=False):
    gameMode=boardpositions.GameMode
    player=boardpositions.CurrentPlayer
    chessBoard=boardpositions.ChessBoard
    finished=False
    boardpositions.setIsPlayerInCheck()
    playerInCheck=boardpositions.IsPlayerInCheck
    if(playerInCheck==True and noOutputMode==False): print("Player",player,"is in CHECK!")
    outOfCheckMoves,moveProbabilities=boardpositions.findAllowedMoves(Sizes,Weights,Biases,debug,noOutputMode)
    if(len(outOfCheckMoves)>0):
        if(len(outOfCheckMoves)<=8):
            #Check if only king is left and opponent can still win!
            if(noOutputMode==False): print("Less than 9 moves available! Check if game is decided.")
            playerSign=boardpositions.CurrentPlayer
            oppSign   =boardpositions.CurrentOpponent
            playerPieces,opponentPieces=[],[]
            for row in chessBoard:
                for col in row:
                    if playerSign*col>0:
                        playerPieces.append(getPieceName(playerSign*col))
                    elif oppSign*col>0:
                        opponentPieces.append(getPieceName(oppSign*col))
            if(noOutputMode==False): print("playerPieces=",playerPieces)
            if(len(playerPieces)==1):
                if(noOutputMode==False): print("opponentPieces=",opponentPieces)
                if(len(opponentPieces)>2 or
                   len(opponentPieces)>1 and ("rook" in opponentPieces or "queen" in opponentPieces)):# or Pawn in opponentPieces)):
                    for rand,move in enumerate(outOfCheckMoves):
                        if move[6]!=0:
                            moveInfoList,captPiece=boardpositions.playMove(move, True, debug, noOutputMode)
                            if(noOutputMode==False): printChessBoard(chessBoard,player,rand,-1,moveInfoList,colored)#moveInfoList,debug)
                            return False
                    boardpositions.setWinner(boardpositions.CurrentOpponent)
                    return True
                elif(len(opponentPieces)==2 and ("bishop" in opponentPieces or "night" in opponentPieces)):
                    boardpositions.setWinner(0)
                    return True
        maxNum=len(outOfCheckMoves)
        s = np.sum(moveProbabilities)
        probNormed = moveProbabilities/s
        helper=np.empty_like(moveProbabilities)
        np.round(moveProbabilities*1000,3,helper)
        prob_rounded=helper.astype(np.int32)
        if(  gameMode=="SelfplayRandomVsRandom" or (gameMode=="SelfplayNetworkVsRandom" and player==-1)):
            rand=random.randint(0,maxNum-1)
            if(noOutputMode==False):
                print("Random mover is moving now!")
                print("maxNum,rand=",maxNum,rand)
            move=outOfCheckMoves[rand]
        elif(gameMode=="SelfplayNetworkVsNetwork" or (gameMode=="SelfplayNetworkVsRandom" and player==1)):
            if(noOutputMode==False): print("Neural network is moving now!")
            #np.random.choice(np.arange(maxNum), p=probNormed)
            #rand=random.randint(0,maxNum-1)
            rand=np.argmax(moveProbabilities)
            move=outOfCheckMoves[rand]
        #else:
        #    rand=random.randint(0,maxNum)
        #    move=outOfCheckMoves[rand]
        if(debug): print("(functions (nextMove)): rand=",rand)
        if noOutputMode==False:printMoves(player,outOfCheckMoves,colored,prob_rounded,debug)
        moveInfoList,captPiece=boardpositions.playMove(move, True, debug, noOutputMode)
        if(noOutputMode==False):
            printChessBoard(chessBoard,player,rand,prob_rounded[rand],moveInfoList,colored)#moveInfoList,debug)
    elif(len(outOfCheckMoves)==0 and playerInCheck==True):
        if(noOutputMode==False): print("Player",player,"is CHECKMATE!")
        boardpositions.setWinner(boardpositions.CurrentOpponent)
        finished=True
    elif(len(outOfCheckMoves)==0 and playerInCheck==False):
        if(noOutputMode==False): print("Player",player,"has no more moves available => Remis.")
        boardpositions.setWinner(0.5)
        finished=True
    return finished

@njit(cache=True)
def printChessBoard(chessBoard, player, rand, prob, moveInfoList=[], colored=False):#moveInfoList=[], debug=False):
    print("###### Board- ######")
    print("## A B C D E F G H #")
    bkgColor="49"
    for j in range(len(chessBoard)):
        ChessBoardString=str(8-j)+" "
        colNum=j
        if player==1: colNum=7-j
        printColor=""
        for i in range(8):
            if colored:
                if  ((8-j)%2==0 and i%2==0) or ((8-j)%2==1 and i%2==1): bkgColor="44"
                elif((8-j)%2==0 and i%2==1) or ((8-j)%2==1 and i%2==0): bkgColor="46"
                if   chessBoard[colNum][i]<0: printColor="\033[0;30;"+bkgColor+"m"
                elif chessBoard[colNum][i]>0: printColor="\033[0;37;"+bkgColor+"m"
                else: printColor="\033[1;37;"+bkgColor+"m"
            pieceUnicode=getPieceUnicode(chessBoard[colNum][i],colored)
            ChessBoardString+=printColor+pieceUnicode+" "
        if(colored):
            if(j==2 and len(moveInfoList)>0):
                ChessBoardString+="\033[1;37;49m "+str(8-j)+"  "+moveInfoList[0]+" (\033[1;32;49m"+str(rand)+","+str(prob)+")"
            elif(j==4 and len(moveInfoList)>1):
                ChessBoardString+="\033[1;37;49m "+str(8-j)+"  "+moveInfoList[1]
            elif(j==6 and len(moveInfoList)>2):
                ChessBoardString+="\033[1;37;49m "+str(8-j)+"  "+moveInfoList[2]
            else:
                ChessBoardString+="\033[1;37;49m "+str(8-j)
        else:
            if(j==2 and len(moveInfoList)>0):
                ChessBoardString+=" "+str(8-j)+"  "+moveInfoList[0]+" ("+str(rand)+","+str(prob)+")"
            elif(j==4 and len(moveInfoList)>1):
                ChessBoardString+=" "+str(8-j)+"  "+moveInfoList[1]
            elif(j==4 and len(moveInfoList)>2):
                ChessBoardString+=" "+str(8-j)+"  "+moveInfoList[2]
            else:
                ChessBoardString+=" "+str(8-j)
        print(ChessBoardString)
    print("# A B C D E F G H ##")
    print("###### -Board ######")

@njit(cache=True)
def getPositions(chessBoard,playerSign):
    pPos=typed.List()
    nPos=typed.List()
    bPos=typed.List()
    rPos=typed.List()
    qPos=typed.List()
    kPos=typed.List()
    for row in range(8):
        for col in range(8):
            if   chessBoard[row][col]==playerSign*6: pPos.append( np.array([col,row], dtype=np.int64) )
            elif chessBoard[row][col]==playerSign*5: nPos.append( np.array([col,row], dtype=np.int64) )
            elif chessBoard[row][col]==playerSign*4: bPos.append( np.array([col,row], dtype=np.int64) )
            elif chessBoard[row][col]==playerSign*3: rPos.append( np.array([col,row], dtype=np.int64) )
            elif chessBoard[row][col]==playerSign*2: qPos.append( np.array([col,row], dtype=np.int64) )
            elif chessBoard[row][col]==playerSign*1: kPos.append( np.array([col,row], dtype=np.int64) )
    return typed.List((kPos,qPos,rPos,bPos,nPos,pPos))
