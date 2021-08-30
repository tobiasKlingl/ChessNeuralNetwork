import numpy as np
import inspect

import functions
from functions import printInfo, printDebug, printError
import numba as nb
from numba.experimental import jitclass
#from numba import jit,njit,types,int64,typed


@nb.njit(cache = True)
def getBasicPieceMoves(piece, piecePos, onlyCaptureMoves = False) -> nb.types.ListType(nb.types.ListType(nb.int64)):
    if piece == "pawn":
        if onlyCaptureMoves == True:
            return [[1,1], [-1,1]]
        elif onlyCaptureMoves == False and piecePos[1] == 1:
            return [[1,1], [-1,1], [0,1], [0,2]]
        elif onlyCaptureMoves == False:
            return [[1,1], [-1,1], [0,1]]
    elif(piece == "knight"):
        return [[1,2], [-1,2], [1,-2], [-1,-2], [2,1], [-2,1], [2,-1], [-2,-1]]
    elif(piece == "bishop"):
        return [[1,1], [-1,1], [1,-1], [-1,-1]]
    elif(piece == "rook"):
        return [[1,0], [-1,0], [ 0,1], [0,-1]]
    elif(piece == "queen" or piece == "king"):
        return [[1,1], [-1,1], [1,-1], [-1,-1], [1,0], [-1,0], [0,1], [0,-1]]
    else:
        printError("getBasicPieceMoves", "Piece ", piece, "unknown")
        raise ValueError("Invalid piece name")


@nb.njit(cache = True)
def checkMove(self, player, opponent, piece, piecePos, delta):
    playerSign = functions.getPlayerSign(player)
    oppSign =    functions.getPlayerSign(opponent)
    #pieceNum =   functions.getPieceNum(piece)
        
    validMove = False
    capturedPiece = 0
    newPos = [piecePos[0] + delta[0], piecePos[1] + delta[1]] # hier evtl tuple drauß machen? 

    if __debug__:
        printDebug("checkMove", piece, ":", piecePosition, "+", delt, "=", newPos)
        
    if(newPos[0] < 0 or newPos[0] > 7 or newPos[1] < 0 or newPos[1] > 7 or (playerSign * chessBoard[newPos[1]][newPos[0]] > 0)):
        if __debug__:
            printDebug("checkMove", "Newpos =", newPos, "for", piece, "is invalid! Return.")
        return newPos, validMove, capturedPiece
    else:
        if piece != "pawn":
            if(oppSign * chessBoard[newPos[1]][newPos[0]] > 0):
                validMove=True
                capturedPiece = abs(chessBoard[newPos[1]][newPos[0]])
            else:
                validMove = True
        elif piece == "pawn":
            if delta == [0, 1] and chessBoard[newPos[1]][newPos[0]] == 0: #allowed normal move
                validMove = True
            elif delta == [0,2]: #normal move (Check also in-between position)
                inBetweenPos = [piecePos[0], piecePos[1]+1]
                if chessBoard[inBetweenPos[1]][inBetweenPos[0]] == 0 and chessBoard[newPos[1]][newPos[0]]==0:
                    validMove=True
            elif delta == [1,1] or delta == [-1,1]: # capture move
                if oppSign * chessBoard[newPos[1]][newPos[0]] > 0:
                    validMove = True
                    capturedPiece = abs(chessBoard[newPos[1]][newPos[0]])
            else:
                printError("checkMove", "Player", player, "'s pawn move from piecePosition =", piecePos, "with delta =", delta, "is neither normal nor capture move! => Not allowed!")
                raise ValueError("Invalid move for piece")

    return newPos, validMove, capturedPiece


@nb.njit(cache = True)
def willPlayerBeInCheck(ChessBoard, move, debug, noOutputMode):
    isPlayerInCheckBeforeMove = ChessBoard.IsPlayerInCheck # muss noch geändert werden
    castlingBeforeMove =        ChessBoard.Castling.copy()
    enPassantBeforeMove =       ChessBoard.EnPassant
   
    moveInfo, captPiece = ChessBoard.playMove(move, False, debug, noOutputMode) # change to simulateMove
    if __debug__:
        ChessBoard.printChessBoard("move")
    ChessBoard.setIsPlayerInCheck()
    willPlayerBeInCheck=ChessBoard.IsPlayerInCheck

    ChessBoard.reverseMove(move, captPiece, castlingBeforeMove, enPassantBeforeMove, isPlayerInCheckBeforeMove)
    if __debug__:
        ChessBoard.printChessBoard("reset")

    return willPlayerBeInCheck




############################################################################################################################
#### jited MoveInformation class ###########################################################################################
############################################################################################################################    

MoveInformationSpecs = [
    ('Piece',  nb.types.string),
    ('OldPos', nb.types.ListType(nb.int64)),
    ('NewPos', nb.types.ListType(nb.int64)),
    ('__MoveID', nb.int64),
    ('CapturedPieceNum', nb.int64),
    ('IsCastlingLeft',   nb.boolean),
    ('IsCastlingRight',  nb.boolean),
    ('IsEnpassantMove',  nb.boolean),
]
@jitclass(MoveInformationSpecs)
class MoveInformation(object):

    def __init__(self, piece, oldPos, newPos, capturedPieceNum) -> None:
        self.Piece = piece
        self.OldPos = piecePos
        self.NewPos = newPos
        self.CapturedPieceNumber = capturedPieceNum #0 == no piece captured
        self.moveID = -1

        
    def setIsCastlingLeft(self, isCastlingLeft):
        self.IsCastlingLeft = isCastlingLeft

        
    def setIsCastlingRight(self, isCastlingRight):
        self.IsCastlingLeft = isCastlingLeft

        
    def setIsEnpassantMove(self, isEnpassantMove):
        self.IsEnpassantMove = isEnpassantMOve

        
    def setMoveID(self):
        testSet = (self.IsCastlingLeft, self.IsCastlingRight, self.IsEnpassantMove)

        if testSet.count(True) > 1:
            printError("More than one entry is greater than 1. TestSet = " + testSet, inspect.stack()[0][3], self.__class__.__name__)                
            raise ValueError("More than one entry is greater than 1")
                
        pieceNumber = functions.getPieceNumber(self.Piece)
        moveIdSet = (pieceNumber, self.PiecePos[0], self.PiecePos[1], newPos[0], newPos[1])
        self.__MoveID = mD[moveIdSet]

        
    def getMoveID(self):
        if self.__moveID == -1:
            printError("self.__MoveID = " + self.__MoveID, inspect.stack()[0][3], self.__class__.__name__)                
            raise ValueError("self.__MoveID is not set correctly")
        else:
            return self.__moveID


############################################################################################################################
#### jited MoveManager class ###############################################################################################
############################################################################################################################
        
MoverSpecs = [
    ('Piece',           nb.types.string),
    ('PiecePos',        nb.types.ListType(nb.int64)), #nb.types.ListType(nb.types.ListType(nb.types.Array(nb.types.int64, 1, 'C')))),
    ('ChessBoard',      nb.types.Array(nb.types.int64, 2, 'C')),
    ('CurrentPlayer',   nb.types.string),
    ('CurrentOpponent', nb.types.string),
    ('Deltas',          nb.types.ListType(nb.types.ListType(nb.int64))),
]
@jitclass(MoverSpecs)
class MoveManager(object):

    def __init__(self, piece, piecePos, chessBoard, currentPlayer) -> None:
        self.Piece = piece
        self.PiecePos = piecePos
        self.ChessBoard = chessBoard
        self.CurrentPlayer = currentPlayer
        self.CurrentOpponent = currentPlayer
        self.Deltas = getBasicPieceMoves(self.Piece, self.PiecePos)
        self.NoOutputMode = False

        
    def setOutputMode(self, noOutputMode):
        self.NoOutputMode = noOutputMode
        
        
    def getAllowedMoves(self) -> None:
        if __debug__:
            printDebug(inspect.stack()[0][3], "Piece =", self.Piece, self.__class__.__name__)

        normalMoves = []
        for delta in self.Deltas:
            validMove = True
            capturedPieceNum = 0
            i=1
            while validMove == True and capturedPieceNum == 0:
                Del = [d*i for d in delta]
                if __debug__:
                    printDebug(inspect.stack()[0][3], "Del =", Del, self.__class__.__name__)

                newPos, validMove, capturedPieceNum = checkMove(self.CurrentPlayer,
                                                                self.CurrentOpponent,
                                                                self.Piece,
                                                                self.PiecePos,
                                                                Del)
                if(validMove == False):
                    break

                if(self.Piece == "pawn" and self.PiecePos[1] == 6 and newPos[1] == 7): #Pawn promotion
                    for pieceNum in range (2,6):                                       #Loop over possible pawn promotions: queen(2), rook(3), bishop(4), knight(5)
                        newPiece = functions.getPieceName(pieceNum)
                        moveObject = MoveInformation(self.Piece, newPiece, self.PiecePos, newPos, capturedPieceNum)
                        moveObject.setIsCastlingLeft(False)
                        moveObject.setIsCastlingRight(False)
                        moveObject.setIsEnpassantMove(False)
                        moveObject.setMoveID()

                        willPlayerBeInCheck = functions.willPlayerBeInCheck(self.ChessBoard, moveObject, self.NoOutputMode)
                        if(willPlayerBeInCheck == False):
                            normalMoves.append(moveObject)
                else:
                    moveObject = MoveInformation(self.Piece, 0, self.PiecePos, newPos, capturedPieceNum) #0 in second index means piece does not change
                    moveObject.setIsCastlingLeft(False)
                    moveObject.setIsCastlingRight(False)
                    moveObject.setIsEnpassantMove(False)
                    moveObject.setMoveID()
                    willPlayerBeInCheck = functions.willPlayerBeInCheck(self.ChessBoard, moveObject, self.NoOutputMode)
                    if(willPlayerBeInCheck==False):
                        normalMoves.append(move)
                if(piece == "pawn" or piece == "knight" or piece == "king"):
                    break
                else:
                    i+=1


############################################################################################################################
#### NON-jited MoveDictionary class ########################################################################################
############################################################################################################################
                    
"""
Create the Move dictionary holding all different possible moves in chess (not considering the type of piece that made the move)
SpecsMoveDictionary = [
    ('MoveDict', nb.types.DictType(nb.types.UniTuple(nb.int64, 5), nb.types.int64)),
    ('MoveList', nb.types.ListType(nb.types.UniTuple(nb.int64, 5))),
    ('MoveID'  , nb.int64),
]
    
@jitclass(SpecsMoveDictionary)
"""
class MoveDictionary(object):
    
    def __init__(self) -> None:
        self.MoveDict = nb.typed.Dict.empty(key_type = nb.types.UniTuple(nb.int64, 5), value_type = nb.int64)
        self.MoveList = nb.typed.List(())
        self.MoveID = 0
        self.initializeMoveDict()

        
    def CPlusPlus(self) -> nb.int64: # C type "i++" function
        self.MoveID += 1
        return self.MoveID - 1

    
    def initializeMoveDict(self) -> None:
        self.MoveDict[(1, 4, 0, 2, 0)] = self.CPlusPlus()  #castling short
        self.MoveDict[(1, 4, 0, 6, 0)] = self.CPlusPlus()  #castling long
        self.MoveList = [(1, 4, 0, 2, 0), (1, 4, 0, 6, 0)] #castling short, castling long

        for col in range(8):
            for row in range(8):
                deltas = getBasicPieceMoves("queen", nb.typed.List((row,col)), False) #np.array([[1,1], [-1,1], [1,-1], [-1,-1], [1,0], [-1,0], [0,1], [0,-1]], dtype=np.int64) #Queen and King moves
                self.appendToList("queen", row, col, deltas)
                deltas = getBasicPieceMoves("knight", nb.typed.List((row,col)), False) #np.array([[1,2], [-1,2], [1,-2], [-1,-2], [2,1], [-2,1], [2,-1], [-2,-1]], dtype=np.int64) #Knight moves
                self.appendToList("knight", row, col, deltas)
                deltas = getBasicPieceMoves("pawn", nb.typed.List((row,col)), False) #np.array([[1,1], [-1,1], [0,1]], dtype=np.int64) #Pawn moves
                self.appendToList("pawn", row, col, deltas)

                
    def appendToList(self, piece, row, col, deltas) -> None: # moveList nicht unbedingt gebraucht, oder?
        piecePos = np.array([row,col], dtype = np.int64)
        for delta in deltas: 
            for i in range(1, 8):
                delt = nb.typed.List([d*i for d in delta])
                newPos = [piecePos[0] + delt[0], piecePos[1] + delt[1]]
                if(newPos[0] < 0 or newPos[0] > 7 or newPos[1] < 0 or newPos[1] > 7): #Out of bounds
                    break
                elif(piece == "pawn"):
                    if(piecePos[1] == 6 and newPos[1] == 7): #Pawn promotions
                        for newPieceNumber in range(2,6):    #Loop over possible promotions: knight(5), bishop(4), rook(3), queen(2)
                            self.MoveList.append((newPieceNumber, piecePos[0], piecePos[1], newPos[0], newPos[1]))
                            self.MoveDict[(newPieceNumber,        piecePos[0], piecePos[1], newPos[0], newPos[1])] = self.CPlusPlus()
                else:
                    self.MoveList.append((0, piecePos[0], piecePos[1], newPos[0], newPos[1]))
                    self.MoveDict[(0, piecePos[0], piecePos[1], newPos[0], newPos[1])] = self.CPlusPlus()
                if(piece == "pawn" or piece == "knight" or piece == "king"):
                    break
                else:
                    i += 1

moveDict = MoveDictionary()
mL = moveDict.MoveList
mD = moveDict.MoveDict

if __name__ == '__main__': main()
