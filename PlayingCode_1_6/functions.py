import numpy as np
import random
from timeit import default_timer as timer
from numba import jit, njit, types, int64, typed


@njit(cache = True)
def printInfo(noOutputMode, *message):
    if not noOutputMode:
        print(" ".join(message))

        
@njit(cache = True)
def printDebug(debugMessage, fName = "", cName = ""):
    if cName == "":
        print("(DEBUG)", fName + ":", debugMessage)
    else:
        print("(DEBUG) [" + cName + "]", fName + ":", debugMessage)

@njit(cache = True)
def printError(errorMessage, fName = "", cName = ""):
    if cName == "":
        print("(ERROR)", fName + ":", errorMessage)
    else:
        print("(ERROR) [" + cName + "]", fName + ":", errorMessage)


@njit(cache = True)
def makePosString(pos) -> types.string:
    string = "".join(["[", str(pos[0]), ", ", str(pos[1]), "]"])
    return string
    
    
@njit(cache = True)
def getPlayerIndex(player) -> int64:
    if(player == "white"):
        return 0
    elif(player == "black"):
        return 1
    else:
        printError(" ".join([player, "unknown"]), fName = "getPlayerIndex")
        raise ValueError("Invalid player name")


@njit(cache = True)
def getPlayerSign(player) -> int64:
    if(player == "white"):
        return 1
    elif(player == "black"):
        return -1
    else:
        printError(" ".join([player, "unknown"]), fName = "getPlayerSign")
        raise ValueError("Invalid player name")
    return -99

    
@njit(cache = True)
def getPieceNum(piece) -> int64:
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
    elif(piece == ""): #needed to handle "non-pawn promotion" moves
        return 0
    else:
        printError(" ".join([piece, "unknown"]), fName = "getPieceNum")
        raise ValueError("Invalid piece name")

    
@njit(cache = True)
def getPieceName(pieceNum) -> types.string:
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
        printError(" ".join([str(pieceNum), "unknown"]), fName = "getPieceName")
        raise ValueError("Invalid piece number")


@njit(cache = True)
def getOpponent(player) -> types.string:
    if(player == "white"):
        return "black"
    elif(player == "black"):
        return "white"
    else:
        printError(" ".join(["invalid argument for player =", player]), fName = "getOpponent")
        return ""


@njit(cache = True)
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side = "right")]


@njit(cache = True)
def evaluatePosition(chessBoard):
    evaluation = 0
    for row in chessBoard:
        for col in row:
            if   abs(col) == 2: evaluation += np.sign(col) * 10
            elif abs(col) == 3: evaluation += np.sign(col) * 5
            elif abs(col) == 4: evaluation += np.sign(col) * 3
            elif abs(col) == 5: evaluation += np.sign(col) * 3
            elif abs(col) == 6: evaluation += np.sign(col) * 1
            
    if __debug__:
        printDebug("".join(["Evaluation = ", evaluation]), fName = "evaluatePosition")

    return evaluation
                

@njit(cache = True)
def getReadablePosition(player, position):
    xPos = position[0]
    yPos = position[1]
    
    if  (xPos == 0): xpos = "A"
    elif(xPos == 1): xpos = "B"
    elif(xPos == 2): xpos = "C"
    elif(xPos == 3): xpos = "D"
    elif(xPos == 4): xpos = "E"
    elif(xPos == 5): xpos = "F"
    elif(xPos == 6): xpos = "G"
    elif(xPos == 7): xpos = "H"
    else:
        xpos = ""
        printError("".join(["pos[0]=", str(xPos), " is unknown!"]), fName = "getReadablePosition")

    if  (player == "white"):
        ypos = str(yPos + 1)
    elif(player == "black"):
        ypos = str(7 - yPos + 1)
    else:
        ypos = ""
        printError("".join(["player = ", player, " is unknown!"]), fName = "getReadablePosition")

    readablePosition = xpos + ypos
    if __debug__:
        printDebug("".join(["Converted position:", makePosString(position), "into:", readablePosition]), fName = "getReadablePosition")

    return readablePosition


@njit(cache = True)
def getPieceUnicode(piece, colored):
    pieceUnicode = " "
    pieceAbb = abs(piece)
    if(colored == True):
        if pieceAbb == 1:   pieceUnicode = '\u265A'
        elif pieceAbb == 2: pieceUnicode = '\u265B'
        elif pieceAbb == 3: pieceUnicode = '\u265C'
        elif pieceAbb == 4: pieceUnicode = '\u265D'
        elif pieceAbb == 5: pieceUnicode = '\u265E'
        elif pieceAbb == 6: pieceUnicode = '\u265F'
    else:
        if piece > 0:
            x = 6
        elif piece < 0:
            x = 0

        if pieceAbb == 1:   pieceUnicode = "".join(chr(9812 + pieceAbb - 1 + x))
        elif pieceAbb == 2: pieceUnicode = "".join(chr(9812 + pieceAbb - 1 + x))
        elif pieceAbb == 3: pieceUnicode = "".join(chr(9812 + pieceAbb - 1 + x))
        elif pieceAbb == 4: pieceUnicode = "".join(chr(9812 + pieceAbb - 1 + x))
        elif pieceAbb == 5: pieceUnicode = "".join(chr(9812 + pieceAbb - 1 + x))
        elif pieceAbb == 6: pieceUnicode = "".join(chr(9812 + pieceAbb - 1 + x))

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

@njit(cache = True)
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

@njit(cache = True)
def relu(z):
    #return np.maximum(z, 0)
    return z * (z>0)

@njit(cache = True)
def moveProbs(boardpositions, allMoves):
    numMoves = len(allMoves)
    """
    Num_layers = len(Sizes)
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
        legitMoveProbs=np.ones(numMoves,dtype=np.float64)
        for i,move in enumerate(outOfCheckMoves):
            moveProbs[i]=allMoveProbabilities[move[10]]
    else:
    """
    moveProbs = np.ones(numMoves, dtype = np.float64)
    return moveProbs


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


@njit(cache = True)
def getPositions(chessBoard, playerSign):
    pPos = [] #typed.List() 
    nPos = [] #typed.List()
    bPos = [] #typed.List()
    rPos = [] #typed.List()
    qPos = [] #typed.List()
    kPos = [] #typed.List()

    for row in range(8):
        for col in range(8):
            if chessBoard[row][col] == playerSign * 6:
                pPos.append(np.array([col, row], dtype = np.int64))
            elif chessBoard[row][col] == playerSign * 5:
                nPos.append(np.array([col, row], dtype = np.int64))
            elif chessBoard[row][col] == playerSign * 4:
                bPos.append(np.array([col, row], dtype = np.int64))
            elif chessBoard[row][col] == playerSign * 3:
                rPos.append(np.array([col, row], dtype = np.int64))
            elif chessBoard[row][col] == playerSign * 2:
                qPos.append(np.array([col, row], dtype = np.int64))
            elif chessBoard[row][col] == playerSign * 1:
                kPos.append(np.array([col, row], dtype = np.int64))

    return [kPos, qPos, rPos, bPos, nPos, pPos]















