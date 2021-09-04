import numpy as np
import random
import math
from timeit import default_timer as timer
from numba import jit, njit, types, int64, float64, typed


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
def getPieceValue(piece) -> float64:
    if(piece == "king"):
        return 0.
    elif(piece == "queen"):
        return 10.
    elif(piece == "rook"):
        return 5.
    elif(piece == "bishop"):
        return 3.
    elif(piece == "knight"):
        return 3.
    elif(piece == "pawn"):
        return 1.
    elif(piece == ""): #needed to handle "non-pawn promotion" moves
        return 0.
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
    elif(pieceNum == 0):
        return ""
    else:
        printError(" ".join([str(pieceNum), "unknown"]), fName = "getPieceName")
        raise ValueError("Invalid piece number")


@njit(cache = True)
def getOpponent(player) -> types.string:
    if player == "white":
        return "black"
    elif player == "black":
        return "white"
    else:
        printError(" ".join(["invalid argument for player =", player]), fName = "getOpponent")
        raise ValueError("Invalid player name")

    
@njit(cache = True)
def floatToString(floatnumber) -> str:
    whole = str(math.floor(floatnumber))
    frac = "0"
    digits = float(floatnumber % 1)
    digitsTimes100 = float(digits) * float(100.0)

    if digitsTimes100 is not None:
        if digitsTimes100 < 10:
            frac = "0" + str(math.floor(digitsTimes100))
        else:
            frac = str(math.floor(digitsTimes100))
        
    stringNumber = whole + "." + frac

    return stringNumber
    
    
@njit(cache = True)
def rand_choice_nb(arr, moves):
    """
    :param moves: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    nMoves = len(moves)

    moveEvaluations = np.ones(nMoves, dtype = np.float64)
    for i, move in enumerate(moves):
        moveEvaluations[i] = move.Evaluation
    
    s = np.sum(moveEvaluations)
    probNormed = moveEvaluations/s
        
    #return 1
    return arr[np.searchsorted(np.cumsum(probNormed), np.random.random(), side = "right")]


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
        raise ValueError("Invalid xPos")
    
    if player == "white":
        ypos = str(yPos + 1)
    elif player == "black":
        ypos = str(7 - yPos + 1)
    else:
        ypos = ""
        printError("".join(["player = ", player, " is unknown!"]), fName = "getReadablePosition")
        raise ValueError("Invalid player name")
    
    readablePosition = xpos + ypos
    if __debug__:
        printDebug(" ".join(["Converted position:", makePosString(position), "into:", readablePosition]), fName = "getReadablePosition")

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


@njit(cache = True)
def createPieceListforPrintOut(readableMoveList, colored = False):
    SortedList = typed.List()
    pawnMoves   = "  PAWN: "
    knightMoves = "  KNIGHT: "
    bishopMoves = "  BISHOP: "
    rookMoves   = "  ROOK: "
    queenMoves  = "  QUEEN: "
    kingMoves   = "  KING: "

    for i, readableMove in enumerate(readableMoveList):
        if colored:
            moveWithID = "".join(["\033[1;34;49m", readableMove[1], "->", readableMove[2], "\033[1;37;49m(\033[1;32;49m", readableMove[3], ",\033[1;37;49m", readableMove[4], "); "])
        else:
            moveWithID = "".join([readableMove[1], "->", readableMove[2], "(", readableMove[3], ",", readableMove[4], "); "])

        if readableMove[0] == "pawn":
            pawnMoves += moveWithID
        elif readableMove[0] == "knight":
            knightMoves += moveWithID
        elif readableMove[0] == "bishop":
            bishopMoves += moveWithID
        elif readableMove[0] == "rook":
            rookMoves += moveWithID
        elif readableMove[0] == "queen":
            queenMoves += moveWithID
        elif readableMove[0] == "king":
            kingMoves += moveWithID
        else:
            printError("Unknown pieceName", fName = "createPieceListforPrintOut")
            raise ValueError("Unknown piece name")
    
    SortedList.append(kingMoves)
    SortedList.append(queenMoves)
    SortedList.append(rookMoves)
    SortedList.append(bishopMoves)
    SortedList.append(knightMoves)
    SortedList.append(pawnMoves)

    return SortedList


@njit(cache = True)
def printMoves(player, allMoves, colored, noOutputMode):
    readableMoveList = typed.List()
    
    for i, move in enumerate(allMoves):
        pos_before = getReadablePosition(player, move.PiecePos)
        pos_after = getReadablePosition( player, move.NewPos)
        if move.IsCastlingLong:
            readableMove = [move.Piece, pos_before, pos_after + "(White CASTLING long)", str(i), floatToString(move.Evaluation)]
            #readableMove = [move.Piece, pos_before, pos_after + "(White CASTLING long)", str(move.MoveID), floatToString(move.Evaluation)]
        elif move.IsCastlingShort:
            readableMove = [move.Piece, pos_before, pos_after + "(Black CASTLING short)", str(i), floatToString(move.Evaluation)]
            #readableMove = [move.Piece, pos_before, pos_after + "(Black CASTLING short)", str(move.MoveID), floatToString(move.Evaluation)]
        else:
            readableMove = [move.Piece, pos_before, pos_after, str(i), floatToString(move.Evaluation)]
            #readableMove = [move.Piece, pos_before, pos_after, str(move.MoveID), floatToString(move.Evaluation)]

        if move.CapturedPieceNum != 0:
            if move.IsEnpassantMove:
                readableMove[2] += "(Piece CAPTURED via en-passant)"
            elif move.Piece == "pawn" and move.NewPiece != "pawn":
                readableMove[2] += "(Piece CAPTURED + promotion: " + move.NewPiece + ")"
            else:
                readableMove[2] += "(Piece CAPTURED)"
        else:
            if move.Piece == "pawn" and move.NewPiece != "pawn":
                readableMove[2] += "(promotion: " + move.NewPiece + ")"
        if __debug__:
            printDebug("readableMove:", fName = "printMoves")
            print(readableMove)

        readableMoveList.append(readableMove)

    SortedLists = createPieceListforPrintOut(readableMoveList, colored)

    if not noOutputMode:
        print("Moves for player", player, ":")
        for list in SortedLists:
            print(list)


@njit(cache = True)
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
def moveProbs(allMoves):
    numMoves = len(allMoves)
    """
    Num_layers = len(Sizes)
    if("SelfplayNetwork" in boardpositions.GameMode):# and boardpositions.BoardInformation.CurrentPlayer==1):
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
    moveEvals = np.ones(numMoves, dtype = np.float64)
    for move, moveEval in zip(allMoves, moveEvals):
        move.Evaluation = moveEval


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


#########################################
### functions for playGames.py script ###
#########################################
@njit(cache = True)
def getOutputVal(gameNumber, plyNumber, initialPlayer, initialOpponent, winner, coloredOutput, noOutputMode):
    if coloredOutput:
        textColor, resetColor = "\033[1;31;49m", "\033[1;37;49m"
    else:
        textColor, resetColor = "", ""

    if winner == initialPlayer:
        won = 1.
        printInfo(noOutputMode, textColor + str(gameNumber) + ": Player", winner, "won in", str(plyNumber), "plys!", resetColor)
    elif winner == initialOpponent:
        won = 0.
        printInfo(noOutputMode, textColor + str(gameNumber) + ": Player", winner, "won in", str(plyNumber), "plys!", resetColor)
    elif winner == "draw":
        won = 0.5
        printInfo(noOutputMode, textColor + str(gameNumber) + ": Game ended remis!", resetColor)
    else:
        won = -99.
        printError(" ".join([str(gameNumber) + ": ERROR: Unknown value for winner! winner =", winner]), fName = "getOutputVal")

    return won
