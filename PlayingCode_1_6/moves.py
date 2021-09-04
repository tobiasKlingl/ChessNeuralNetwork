import numpy as np
import inspect

import functions as fn
from functions import printInfo, printDebug, printError, makePosString
import numba as nb
from numba.experimental import jitclass
#from numba import jit,njit,types,int64,typed


@nb.njit(cache = True)
def getBasicPieceMoves(piece, piecePos, onlyCaptureMoves = False) -> nb.types.ListType(nb.types.Array(nb.types.int64, 1, 'C')):
    if piece == "pawn":
        if onlyCaptureMoves == True:
            return nb.typed.List((np.array([1,1]), np.array([-1,1])))
        elif onlyCaptureMoves == False and piecePos[1] == 1:
            return nb.typed.List((np.array([1,1]), np.array([-1,1]), np.array([0,1]), np.array([0,2])))
        else:
            return nb.typed.List((np.array([1,1]), np.array([-1,1]), np.array([0,1])))
    elif(piece == "knight"):
        return nb.typed.List((np.array([1,2]), np.array([-1,2]), np.array([1,-2]), np.array([-1,-2]),
                              np.array([2,1]), np.array([-2,1]), np.array([2,-1]), np.array([-2,-1])))
    elif(piece == "bishop"):
        return nb.typed.List((np.array([1,1]), np.array([-1,1]), np.array([1,-1]), np.array([-1,-1])))
    elif(piece == "rook"):
        return nb.typed.List((np.array([1,0]), np.array([-1,0]), np.array([ 0,1]), np.array([0,-1])))
    elif(piece == "queen" or piece == "king"):
        return nb.typed.List((np.array([1,1]), np.array([-1,1]), np.array([1,-1]), np.array([-1,-1]),
                              np.array([1,0]), np.array([-1,0]), np.array([0, 1]), np.array([0,-1])))
    else:
        errorMessage = "".join(["Piece ", piece, " is unknown"])
        printError(errorMessage, fName = "getBasicPieceMoves")
        return nb.typed.List((np.array([-99,-99]), np.array([-99,-99])))

    
@nb.njit(cache = True)
def checkMove(chessBoard, piece, piecePos, delta):
    playerSign = fn.getPlayerSign(chessBoard.BoardInfo.CurrentPlayer)
    oppSign =    fn.getPlayerSign(chessBoard.BoardInfo.CurrentOpponent)
    validMove = False
    capturedPiece = 0
    newPos = piecePos + delta

    if __debug__:
        debugMessage = "".join([piece, ": ", makePosString(piecePos), " + ", makePosString(delta), " = ", makePosString(newPos)])
        printDebug(debugMessage, fName = "checkMove")

    if newPos[0] < 0 or newPos[0] > 7 or newPos[1] < 0 or newPos[1] > 7 or (playerSign * chessBoard.Board[newPos[1]][newPos[0]] > 0):
        if __debug__:
            debugMessage = "".join(["Newpos = ", makePosString(newPos), " for ", piece, " is invalid! Return."])
            printDebug(debugMessage, fName = "checkMove")
        return newPos, validMove, capturedPiece
    else:
        if piece != "pawn":
            validMove = True
            if oppSign * chessBoard.Board[newPos[1]][newPos[0]] > 0:
                capturedPiece = abs(chessBoard.Board[newPos[1]][newPos[0]])
        elif piece == "pawn":
            if (delta == np.array([0, 1])).all(): #normal move
                if chessBoard.Board[newPos[1]][newPos[0]] == 0: 
                    validMove = True
                else:
                    validMove = False
            elif (delta == np.array([0, 2])).all(): #normal move (Check also in-between position)
                inBetweenPos = np.array([piecePos[0], piecePos[1] + 1])
                if chessBoard.Board[inBetweenPos[1]][inBetweenPos[0]] == 0 and chessBoard.Board[newPos[1]][newPos[0]] == 0:
                    validMove = True
                else:
                    validMove = False
            elif (delta == np.array([1, 1])).all() or (delta == np.array([-1, 1])).all(): # capture move
                if oppSign * chessBoard.Board[newPos[1]][newPos[0]] > 0:
                    validMove = True
                    capturedPiece = abs(chessBoard.Board[newPos[1]][newPos[0]])
            else:
                errorMessage = "".join([chessBoard.BoardInfo.CurrentPlayer, "'s pawn move from piecePos = ", makePosString(piecePos), " with delta = ", makePosString(delta), " is neither normal nor capture move!"])
                printError(errorMessage, fName = "checkMove")
                raise ValueError("Invalid move for piece")

    return newPos, validMove, capturedPiece


@nb.njit(cache = True)
def printMoveInfo(move):
    print("move.Piece =",    move.Piece, "\n",
          "move.NewPiece =", move.NewPiece, "\n",
          "move.PiecePos =", move.PiecePos, "\n",
          "move.NewPos =",   move.NewPos, "\n",
          "move.MoveID =",   move.MoveID, "\n",
          "move.Evaluation =",      move.Evaluation, "\n",
          "move.CapturedPieceNum=", move.CapturedPieceNum, "\n",
          "move.IsCastlingLong  =", move.IsCastlingLong, "\n",
          "move.IsCastlingShort =", move.IsCastlingShort, "\n",
          "move.IsEnpassantMove =", move.IsEnpassantMove, "\n",
          "move.IsPromotionMove =", move.IsPromotionMove)
    

@nb.njit(cache = True)
def willPlayerBeInCheck(chessBoard, move, noOutputMode) -> nb.types.boolean:
    isPlayerInCheckBeforeMove = chessBoard.BoardInfo.IsPlayerInCheck # muss noch geändert werden

    chessBoard.simulateMove(move)

    if __debug__:
        printMoveInfo(move)
        chessBoard.printChessBoard("move")

    chessBoard.setIsPlayerInCheck()
    move.Evaluation = chessBoard.evaluate()
    willPlayerBeInCheck = chessBoard.BoardInfo.IsPlayerInCheck

    chessBoard.reverseSimulation(move, isPlayerInCheckBeforeMove)
    if __debug__:
        chessBoard.printChessBoard("reset")

    return willPlayerBeInCheck


############################################################################################################################
#### jited MoveInformation class ###########################################################################################
############################################################################################################################    

MoveInformationSpecs = [
    ('ClassName',        nb.types.string),
    ('Piece',            nb.types.string),
    ('NewPiece',         nb.types.string),
    ('PiecePos',         nb.types.Array(nb.types.int64, 1, 'C')), #nb.types.ListType(nb.int64)),
    ('NewPos',           nb.types.Array(nb.types.int64, 1, 'C')), #nb.types.ListType(nb.int64)),
    ('MoveID',           nb.int64),
    ('Evaluation',   nb.float64),
    ('CapturedPieceNum', nb.int64),
    ('IsCastlingLong',   nb.boolean),
    ('IsCastlingShort',  nb.boolean),
    ('IsEnpassantMove',  nb.boolean),
    ('IsPromotionMove',  nb.boolean),
]

@jitclass(MoveInformationSpecs)
class MoveInformation(object):

    def __init__(self, piece, newPiece, piecePos, newPos, capturedPieceNum) -> None:
        self.ClassName = "MoveInformation"
        self.Piece =            piece
        self.NewPiece =         newPiece
        self.PiecePos =         piecePos
        self.NewPos =           newPos
        self.MoveID =           -1
        self.Evaluation =       -1.0
        self.CapturedPieceNum = capturedPieceNum #0 == no piece captured
        self.IsCastlingLong =   False
        self.IsCastlingShort =  False
        self.IsEnpassantMove =  False
        self.IsPromotionMove =  False
        

    def setIsCastlingLong(self, isCastlingLong):
        self.IsCastlingLong = isCastlingLong

        
    def setIsCastlingShort(self, isCastlingShort):
        self.IsCastlingShort = isCastlingShort

        
    def setIsEnpassantMove(self, isEnpassantMove):
        self.IsEnpassantMove = isEnpassantMove

        
    def setIsPromotionMove(self, isPromotionMove):
        self.IsPromotionMove = isPromotionMove

        
    def checkSpecialMovesSetting(self):
        items = [self.IsCastlingLong, self.IsCastlingShort, self.IsEnpassantMove, self.IsPromotionMove]

        if items.count(True) > 1:
            printError(" ".join(["Several of the following items are set to true:"]), fName = "playMove", cName = self.ClassName) 
            print("IsCastlingLong =",  self.IsCastlingLong)
            print("IsEnpassantMove =", self.IsEnpassantMove)
            print("IsCastlingShort =", self.IsCastlingShort)
            print("IsPromotionMove =", self.IsPromotionMove)
            raise ValueError("More than one item is true")

        
    def setMoveID(self, moveDict):
        testList = [self.IsCastlingLong, self.IsCastlingShort, self.IsEnpassantMove]

        if testList.count(True) > 1:
            printError("More than one entry is greater than 1. testList :", fName = "setMoveID", cName = self.ClassName)
            print("[self.IsCastlingLong, self.IsCastlingShort, self.IsEnpassantMove] = ", testList)
            raise ValueError("More than one entry is greater than 1")

        startNum = 0
        if not self.IsCastlingLong and not self.IsCastlingShort:

            if self.Piece != self.NewPiece:
                startNum = fn.getPieceNum(self.NewPiece)
        else:
            startNum = fn.getPieceNum("king")
            
        moveString = "".join([str(startNum), str(self.PiecePos[0]), str(self.PiecePos[1]), str(self.NewPos[0]), str(self.NewPos[1])])

        if __debug__:
            printDebug(" ".join(["moveString =", moveString]), fName = "setMoveID", cName = self.ClassName)

        self.MoveID = moveDict[moveString]

    
    def getMoveID(self):
        if self.MoveID == -1:
            printError("".join(["self.MoveID =", str(self.MoveID)]), fName = "getMoveID", cName = self.ClassName)
            raise ValueError("self.MoveID is not set correctly")

        return self.MoveID


############################################################################################################################
#### jited MoveManager class ###############################################################################################
############################################################################################################################
        
MoverSpecs = [
    ('ClassName',       nb.types.string),
    ('Piece',           nb.types.string),
    ('PiecePos',        nb.types.Array(nb.types.int64, 1, 'C')),
    ('Deltas',          nb.types.ListType(nb.types.Array(nb.types.int64, 1, 'C'))),
    ('NoOutputMode',    nb.boolean),
]
@jitclass(MoverSpecs)
class MoveManager(object):

    def __init__(self, piece, piecePos, noOutputMode) -> None:
        self.ClassName = "MoveManager"
        self.Piece = piece
        self.PiecePos = piecePos
        self.Deltas = nb.typed.List(getBasicPieceMoves(self.Piece, self.PiecePos))
        self.NoOutputMode = noOutputMode
        
        
    def findNormalMovesFromPiecePos(self, chessBoard, moveDict) -> list:
        if __debug__:
            printDebug("".join(["Piece = ", self.Piece]), fName = "findNormalMovesFromPiecePos", cName = self.ClassName)

        normalMoves = []
        
        for delta in self.Deltas:
            validMove = True
            capturedPieceNum = 0
            i = 1
            while validMove and capturedPieceNum == 0:
                Del = delta * i #[d*i for d in delta]
                if __debug__:
                    printDebug("".join(["Del = ", makePosString(Del)]), fName = "findNormalMovesFromPiecePos", cName = self.ClassName)
                               
                newPos, validMove, capturedPieceNum = checkMove(chessBoard, self.Piece, self.PiecePos, Del)

                if not validMove:
                    break

                if(self.Piece == "pawn" and self.PiecePos[1] == 6 and newPos[1] == 7): #Pawn promotion
                    pawnPromotionList = ["queen", "rook", "bishop", "knight"]
                    for promotedPiece in pawnPromotionList:
                        moveObject = MoveInformation(self.Piece, promotedPiece, self.PiecePos, newPos, capturedPieceNum)
                        moveObject.setMoveID(moveDict)
                        moveObject.setIsPromotionMove(True)

                        if not willPlayerBeInCheck(chessBoard, moveObject, self.NoOutputMode):
                            print("moveObject.Evaluation =", moveObject.Evaluation)
                            normalMoves.append(moveObject)
                else:
                    moveObject = MoveInformation(self.Piece, self.Piece, self.PiecePos, newPos, capturedPieceNum) #0 in 2nd index -> move is possible for different types of piece (e.g queen and bishop)
                    moveObject.setMoveID(moveDict)
                    if not willPlayerBeInCheck(chessBoard, moveObject, self.NoOutputMode):
                       normalMoves.append(moveObject)

                if self.Piece == "pawn" or self.Piece == "knight" or self.Piece == "king":
                    break
                else:
                    i+=1

        return normalMoves

    
    def findCastlingMoves(self, chessBoard, moveDict) -> list:
        if __debug__:
            printDebug("".join(["Piece = ", self.Piece]), fName = "findCastlingMoves", cName = self.ClassName)

        castlingMoves = []
        capturedPieceNum = 0

        if(chessBoard.BoardInfo.Castling[chessBoard.BoardInfo.CurrentPlayer+",long"] == 1 and chessBoard.Board[0][1] == 0 and chessBoard.Board[0][2] == 0 and chessBoard.Board[0][3] == 0):
            if __debug__:
                debugMessage = "".join(["Adding castling LONG to player ", chessBoard.BoardInfo.CurrentPlayer,"'s castlingMoves."])
                printDebug(debugMessage, fName = "getCastlingMoves", cName = self.ClassName)

            newPos =       np.array([2, 0])
            inBetweenPos = np.array([3, 0])
            moveObject =          MoveInformation(self.Piece, "king", self.PiecePos, newPos,       capturedPieceNum)
            inBetweenMoveObject = MoveInformation(self.Piece, "king", self.PiecePos, inBetweenPos, capturedPieceNum)

            helper_willPlayerBeInCheck = False
            for i, moveOb in enumerate([moveObject, inBetweenMoveObject]):
                moveOb.setIsCastlingLong(True)
                if i == 0:
                    moveOb.setMoveID(moveDict)
                if willPlayerBeInCheck(chessBoard, moveOb, self.NoOutputMode): 
                    helper_willPlayerBeInCheck = True

            if not helper_willPlayerBeInCheck:
                castlingMoves.append(moveObject)

        if(chessBoard.BoardInfo.Castling[chessBoard.BoardInfo.CurrentPlayer+",short"] == 1 and chessBoard.Board[0][6] == 0 and chessBoard.Board[0][5] == 0):
            if __debug__:
                debugMessage = "".join(["Adding castling SHORT to player ", chessBoard.BoardInfo.CurrentPlayer,"'s castlingMoves."])
                printDebug(debugMessage, fName = "getCastlingMoves", cName = self.ClassName)

            newPos =       np.array([6, 0])
            inBetweenPos = np.array([5, 0])
            moveObject =          MoveInformation(self.Piece, "king", self.PiecePos, newPos,       capturedPieceNum)
            inBetweenMoveObject = MoveInformation(self.Piece, "king", self.PiecePos, inBetweenPos, capturedPieceNum)

            helper_willPlayerBeInCheck = False
            for i, moveOb in enumerate([moveObject, inBetweenMoveObject]):
                moveOb.setIsCastlingShort(True)
                if i == 0:
                    moveOb.setMoveID(moveDict)
                if willPlayerBeInCheck(chessBoard, moveOb, self.NoOutputMode): 
                    helper_willPlayerBeInCheck = True

            if not helper_willPlayerBeInCheck:
                castlingMoves.append(moveObject)

        return castlingMoves


    def findEnPassantMoves(self, chessBoard, moveDict) -> list:
        if __debug__:
            debugMessage = " ".join(["En-passant move available for player", chessBoard.BoardInfo.CurrentPlayer, "'s pawn at", makePosString(self.PiecePos)])
            printDebug(debugMessage, fName = "findEnPassantMoves", cName = self.ClassName)

        enPassantMoves = []
        newPos = np.array([chessBoard.BoardInfo.EnPassant, 5])
        capturedPieceNum = 6 #only pawns can be captured via en-passant 
        
        moveObject = MoveInformation(self.Piece, self.Piece, self.PiecePos, newPos, capturedPieceNum) # vielleicht piece = 6 hier?
        moveObject.setIsEnpassantMove(True)
        moveObject.setMoveID(moveDict)

        if not willPlayerBeInCheck(chessBoard, moveObject, self.NoOutputMode):
            enPassantMoves.append(moveObject)

        return enPassantMoves


@nb.njit(cache = True)
def appendToList(moveDict, piece, piecePos, deltas, moveID):
    for delta in deltas: 
        for i in range(1, 8):
            delt = delta * i
            newPos = piecePos + delt
            if(newPos[0] < 0 or newPos[0] > 7 or newPos[1] < 0 or newPos[1] > 7): #Out of bounds
                break
            elif(piece == "pawn"):
                if(piecePos[1] == 6 and newPos[1] == 7): #Pawn promotions
                    for newPieceNumber in range(2, 6):   #Loop over possible promotions: knight(5), bishop(4), rook(3), queen(2)
                        moveString = "".join([str(newPieceNumber), str(piecePos[0]), str(piecePos[1]), str(newPos[0]), str(newPos[1])])
                        moveDict[moveString] = moveID
                        moveID += 1 
            else:
                moveString = "".join(["0", str(piecePos[0]), str(piecePos[1]), str(newPos[0]), str(newPos[1])])
                moveDict[moveString] = moveID
                moveID += 1 
            if(piece == "pawn" or piece == "knight" or piece == "king"):
                break

    return moveID


@nb.njit(cache = True)
def getMoveDict() -> nb.typed.Dict:
    moveDict = nb.typed.Dict.empty(nb.types.string, nb.int64)
    moveID = 0
    moveDict["14020"] = moveID #castling short
    moveID += 1
    moveDict["14060"] = moveID #castling long
    moveID += 1

    for col in range(8):
        for row in range(8):
            piecePos = np.array([row,col], dtype = np.int64)
            for piece in ["queen", "knight", "pawn"]:
                deltas = getBasicPieceMoves(piece, piecePos, False)
                moveID = appendToList(moveDict, piece, piecePos, deltas, moveID)

    return moveDict
