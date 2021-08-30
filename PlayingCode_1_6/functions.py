import numpy as np
import random
from timeit import default_timer as timer
from numba import jit,njit,types,int64,typed


@njit(cache = True)
def printInfo(noOutputMode, *message):
    if not noOutputMode:
        print(" ".join(message))

        
@njit(cache = True)
def printDebug(fName, *debugMessage, cName = ""):
    print("(DEBUG)", cName, fName, " ".join(debugMessage))


@njit(cache = True)
def printError(fName, *errorMessage, cName = ""):
    print("(ERROR)", cName, fName, " ".join(errorMessage))
    
    
@njit(cache = True)
def getPlayerSign(player) -> None:
    if(player == "white"):
        return 1
    elif(player == "black"):
        return -1
    else:
        printError(inspect.stack()[0][3], player, "unknown")
        raise ValueError("Invalid player name")

    
@njit(cache = True)
def getPieceNumber(piece):
    if(piece == "king"):
        return 1
    elif(piece == "queen"):
        return 2
    elif(piece == "rook"):
        return 3
    elif(piece == "bishop"):
        return 4
    elif(piece == "knight"):
        return 5
    elif(piece == "pawn"):
        return 6
    else:
        printError(inspect.stack()[0][3], piece, "unknown!")
        raise ValueError("Invalid piece name")

    
@njit(cache = True)
def getPieceName(pieceNum):
    if(pieceNum == 1):
        return "king"
    elif(pieceNum == 2):
        return "queen"
    elif(pieceNum == 3):
        return "rook"
    elif(pieceNum == 4):
        return "bishop"
    elif(pieceNum == 5):
        return "knight"
    elif(pieceNum == 6):
        return "pawn"
    else:
        printError(inspect.stack()[0][3], pieceNum, "unknown!")
        raise ValueError("Invalid piece number")


@njit(cache = True)
def getOpponent(player):
    if(player == +1):
        opponent=-1
    elif(player == -1):
        opponent=+1
    else:
        print("(ERROR (function (getOpponent): invalid argument for player=",player)
    return opponent


@njit(cache = True)
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


@njit(cache=True)
def evaluatePosition(chessBoard):
    evaluation=0
    for row in chessBoard:
        for col in row:
            if   abs(col)==2: evaluation+=np.sign(col)*10
            elif abs(col)==3: evaluation+=np.sign(col)*5
            elif abs(col)==4: evaluation+=np.sign(col)*3
            elif abs(col)==5: evaluation+=np.sign(col)*3
            elif abs(col)==6: evaluation+=np.sign(col)*1
    print("functions (evaluatePosition): Evaluation=",evaluation)
    return evaluation
                
"""
@njit(cache=True)
def getMoveID(move,debug,noOutputMode):
    pieceBefore=move[0]
    pieceAfter =move[1]
    beforeX=move[2]
    beforeY=move[3]
    afterX =move[4]
    afterY =move[5]
    delX=afterX-beforeX
    delY=afterY-beforeY

    if pieceBefore==1 or pieceBefore==2: #rookMoveIDs
        if delY==0: # horizontal
            startID=0
            rowID=56*beforeY
            if   beforeX>afterX:
                moveID=startID + rowID + 8*beforeX + delX
            elif beforeX<afterX: 
                moveID=startID + rowID + 8*beforeX + delX - 1
        elif delX==0: # vertical
            startID=448 #56*8
            colID=56*beforeX
            if   beforeY>afterY:
                moveID=startID + colID + 8*beforeY + delY
            elif beforeY<afterY: 
                moveID=startID + colID + 8*beforeY + delY - 1
    elif pieceBefore==1 or pieceBefore==3: #bishopMoveIDs
        DELT_XY=beforeX-beforeY
        if DELT_XY==0: # On main pos diagonals
            startID=896 #56*8*2
            if   delX<0 and delY<0: moveID=startID + 8*beforeX + delX
            elif delX>0 and delY>0: moveID=startID + 8*beforeX + delX -1
            elif (delX>0 and delY<0) or (delX<0 and delY>0):
                if delX>0 and delY<0: 
                    startID=951 #896+55
                    delt=delX
                elif delX<0 and delY>0:
                    startID=963 #896+55+12
                    delt=delY
                if beforeX==1:   moveID=startID + delt     #952    // #964    
                elif beforeX==2: moveID=startID + delt + 1 #953,954// #965,966
                elif beforeX==3: moveID=startID + delt + 3 #955-957// #967-969
                elif beforeX==4: moveID=startID + delt + 6 #958-960// #970-972
                elif beforeX==5: moveID=startID + delt + 9 #961,962// #973,974
                elif beforeX==6: moveID=startID + delt + 11#963    // #975    
        elif DELT_XY==1: # shifted +1 main diagonal
            startID=975  #896+7+9+11+13+13+11+9+7
        elif DELT_XY==2: # shifted +2 main diagonal
            startID=1042 #975+7+9+11+13+11+9+7
        elif DELT_XY==3: # shifted +3 main diagonal
            startID=1096 #1042+7+9+11+11+9+7
        elif DELT_XY==4: # shifted +4 main diagonal
            startID=1139 #1096+7+9+11+9+7
        elif DELT_XY==5: # shifted +5 main diagonal
            startID=1171 #1139+7+9+9+7
        elif DELT_XY==6: # shifted +6 main diagonal
            startID=1194 #1171+7+9+7
        elif DELT_XY==7: # shifted +7 main diagonal
            startID=1201 #1194+7
        if   delX<0 and delY<0: moveID=startID + (8-DELT_XY)*(beforeX-DELT_XY) + delX
        elif delX>0 and delY>0: moveID=startID + (8-DELT_XY)*(beforeX-DELT_XY) + delX -1
        elif (delX>0 and delY<0) or (delX<0 and delY>0):
            if delX>0 and delY<0: 
                startID=951 #896+55
                delt=delX
            elif delX<0 and delY>0:
                startID=963 #896+55+12
                delt=delY
            if beforeX==1:   moveID=startID + delt     #952    // #964    
            elif beforeX==2: moveID=startID + delt + 1 #953,954// #965,966
            elif beforeX==3: moveID=startID + delt + 3 #955-957// #967-969
            elif beforeX==4: moveID=startID + delt + 6 #958-960// #970-972
            elif beforeX==5: moveID=startID + delt + 9 #961,962// #973,974
            elif beforeX==6: moveID=startID + delt + 11#963    // #975    
"""

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
    for i,readableMove in enumerate(readableMoveList):
        if colored:
            moveWithID="\033[1;34;49m"+readableMove[1]+"->"+readableMove[2]+"\033[1;37;49m(\033[1;32;49m"+readableMove[3]+",\033[1;37;49m"+str(prob[i])+") ; "
        else:
            moveWithID=readableMove[1]+"->"+readableMove[2]+"("+readableMove[3]+","+str(prob[i])+") ; "
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
        moveID=move[10]
        pos_before=getReadablePosition(player,move[2],move[3],debug)
        pos_after=getReadablePosition( player,move[4],move[5],debug)
        if(castlingL==True):
            readableMove=[pieceName,pos_before,pos_after+"(White CASTLING long)",str(move[10])]
        elif(castlingS==True):
            readableMove=[pieceName,pos_before,pos_after+"(Black CASTLING short)",str(move[10])]
        else:
            readableMove=[pieceName,pos_before,pos_after,str(move[10])]
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
def getNNInput(boardpositions, move, debug, noOutputMode):

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
def moveProbs(boardpositions,Sizes,Weights,Biases,outOfCheckMoves,nnInp,debug):
    Num_layers=len(Sizes)
    nMoves=len(outOfCheckMoves)
    if("SelfplayNetwork" in boardpositions.GameMode):# and boardpositions.CurrentPlayer==1):
        #a=np.ascontiguousarray(nnInput[:lenInput].transpose())
        a=nnInp.reshape(780,1)
        for i,(b,w) in enumerate(zip(Biases, Weights)):
            print("a.shape=",a.shape)
            print("w.shape=",w.shape)
            print("b.shape=",b.shape)
            if i==Num_layers-2:
                a = sigmoid(w@a + b)
            else:
                a = relu(w@a + b)
        allMoveProbabilities=a.reshape(-1)
        if(debug):
            print("All probabilities (including unallowed moves):",allMoveProbabilities)
        legitMoveProbs=np.ones(nMoves,dtype=np.float64)
        for i,move in enumerate(outOfCheckMoves):
            legitMoveProbs[i]=allMoveProbabilities[move[10]]
    else:
        legitMoveProbs=np.ones(nMoves,dtype=np.float64)
    return legitMoveProbs

@jit
def nextMove(boardpositions,boardInput,mD,Sizes,Weights,Biases,colored=False,debug=False,noOutputMode=False):
    gameMode=boardpositions.GameMode
    player=boardpositions.CurrentPlayer
    chessBoard=boardpositions.ChessBoard
    boardpositions.setIsPlayerInCheck()
    playerInCheck=boardpositions.IsPlayerInCheck
    if(playerInCheck==True and noOutputMode==False): print("Player",player,"is in CHECK!")
    outOfCheckMoves,moveProbabilities=boardpositions.findAllowedMoves(boardInput,mD,Sizes,Weights,Biases,debug,noOutputMode)
    finished=False
    moveID=-1
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
                            if(noOutputMode==False): printChessBoard(chessBoard,player,move[10],-1,moveInfoList,colored)#moveInfoList,debug)
                            return False,-1
                    boardpositions.setWinner(boardpositions.CurrentOpponent)
                    return True,-1
                elif(len(opponentPieces)==2 and ("bishop" in opponentPieces or "night" in opponentPieces)):
                    boardpositions.setWinner(0)
                    return True,-1
        maxNum=len(outOfCheckMoves)
        argMax=np.argmax(moveProbabilities)
        maxProb=moveProbabilities[argMax]
        helper=np.empty_like(moveProbabilities)
        s = np.sum(moveProbabilities)
        probNormed = moveProbabilities/s
        np.round(moveProbabilities*1000,3,helper)
        prob_rounded=helper.astype(np.int32)
        if(  gameMode=="SelfplayRandomVsRandom" or (gameMode=="SelfplayNetworkVsRandom" and player==-1)):
            rand=random.randint(0,maxNum-1)
            move=outOfCheckMoves[rand]
            if(noOutputMode==False):
                print("Random mover is moving now!")
                print("maxNum,rand=",maxNum,rand)
                print("move[10]=",move[10])
            moveID=move[10]
        elif(gameMode=="SelfplayNetworkVsNetwork" or (gameMode=="SelfplayNetworkVsRandom" and player==1)):
            #rand=np.argmax(moveProbabilities)
            rand=rand_choice_nb(np.arange(maxNum),probNormed)
            move=outOfCheckMoves[rand]
            if(noOutputMode==False):
                print("Neural network is moving now!")
                print("maxNum,rand=",maxNum,rand)
                print("move[10]=",move[10])
            moveID=move[10]
            #np.random.choice(np.arange(maxNum), p=probNormed)
            #rand=random.randint(0,maxNum-1)
        #else:
        #    rand=random.randint(0,maxNum)
        #    move=outOfCheckMoves[rand]
        #    moveID=move[10]
        if(debug): print("(functions (nextMove)): rand=",rand)
        if noOutputMode==False:printMoves(player,outOfCheckMoves,colored,prob_rounded,debug)
        moveInfoList,captPiece=boardpositions.playMove(move, True, debug, noOutputMode)
        if(noOutputMode==False):
            printChessBoard(chessBoard,player,move[10],prob_rounded[rand],moveInfoList,colored)#moveInfoList,debug)
            print("boardpositions.Castling=",boardpositions.Castling)
            print("boardpositions.EnPassant=",boardpositions.EnPassant)
    elif(len(outOfCheckMoves)==0 and playerInCheck==True):
        if(noOutputMode==False): print("Player",player,"is CHECKMATE!")
        boardpositions.setWinner(boardpositions.CurrentOpponent)
        finished=True
    elif(len(outOfCheckMoves)==0 and playerInCheck==False):
        if(noOutputMode==False): print("Player",player,"has no more moves available => Remis.")
        boardpositions.setWinner(0.5)
        finished=True
    posEval=evaluatePosition(boardpositions.ChessBoard)
    return finished,moveID,posEval


@njit(cache=True)
def printChessBoard(chessBoard, player, moveID, prob, moveInfoList=[], colored=False):#moveInfoList=[], debug=False):
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
                ChessBoardString+="\033[1;37;49m "+str(8-j)+"  "+moveInfoList[0]+" (\033[1;32;49m"+str(moveID)+","+str(prob)+")"
            elif(j==4 and len(moveInfoList)>1):
                ChessBoardString+="\033[1;37;49m "+str(8-j)+"  "+moveInfoList[1]
            elif(j==6 and len(moveInfoList)>2):
                ChessBoardString+="\033[1;37;49m "+str(8-j)+"  "+moveInfoList[2]
            else:
                ChessBoardString+="\033[1;37;49m "+str(8-j)
        else:
            if(j==2 and len(moveInfoList)>0):
                ChessBoardString+=" "+str(8-j)+"  "+moveInfoList[0]+" ("+str(moveID)+","+str(prob)+")"
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
