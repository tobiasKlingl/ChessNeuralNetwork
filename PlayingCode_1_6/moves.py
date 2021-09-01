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
            return [np.array([1,1]), np.array([-1,1])]
        elif onlyCaptureMoves == False and piecePos[1] == 1:
            return [np.array([1,1]), np.array([-1,1]), np.array([0,1]), np.array([0,2])]
        else:
            return [np.array([1,1]), np.array([-1,1]), np.array([0,1])]
    elif(piece == "knight"):
        return [np.array([1,2]), np.array([-1,2]), np.array([1,-2]), np.array([-1,-2]),
                np.array([2,1]), np.array([-2,1]), np.array([2,-1]), np.array([-2,-1])]
    elif(piece == "bishop"):
        return [np.array([1,1]), np.array([-1,1]), np.array([1,-1]), np.array([-1,-1])]
    elif(piece == "rook"):
        return [np.array([1,0]), np.array([-1,0]), np.array([ 0,1]), np.array([0,-1])]
    elif(piece == "queen" or piece == "king"):
        return [np.array([1,1]), np.array([-1,1]), np.array([1,-1]), np.array([-1,-1]),
                np.array([1,0]), np.array([-1,0]), np.array([0, 1]), np.array([0,-1])]
    else:
        errorMessage = "".join(["Piece ", piece, " is unknown"])
        printError(errorMessage, fName = "getBasicPieceMoves")
        return [np.array([-99,-99]), np.array([-99,-99])]

    
@nb.njit(cache = True)
def checkMove(player, opponent, chessBoard, piece, piecePos, delta):
    playerSign = fn.getPlayerSign(player)
    oppSign =    fn.getPlayerSign(opponent)
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
            if(oppSign * chessBoard.Board[newPos[1]][newPos[0]] > 0):
                validMove=True
                capturedPiece = abs(chessBoard.Board[newPos[1]][newPos[0]])
            else:
                validMove = True
        elif piece == "pawn":
            if (delta == np.array([0, 1])).all() and chessBoard.Board[newPos[1]][newPos[0]] == 0: #allowed normal move
                validMove = True
            elif (delta == np.array([0,2])).all(): #normal move (Check also in-between position)
                inBetweenPos = [piecePos[0], piecePos[1]+1]
                if chessBoard.Board[inBetweenPos[1]][inBetweenPos[0]] == 0 and chessBoard.Board[newPos[1]][newPos[0]]==0:
                    validMove=True
            elif (delta == np.array([1,1])).all() or (delta == np.array([-1,1])).all(): # capture move
                if oppSign * chessBoard.Board[newPos[1]][newPos[0]] > 0:
                    validMove = True
                    capturedPiece = abs(chessBoard.Board[newPos[1]][newPos[0]])
            else:
                errorMessage = "".join([player, "'s pawn move from piecePos = ", makePosString(piecePos), " with delta = ", makePosString(delta), " is neither normal nor capture move!"])
                printError(errorMessage, fName = "checkMove")
                raise ValueError("Invalid move for piece")

    return newPos, validMove, capturedPiece


@nb.njit(cache = True)
def willPlayerBeInCheck(chessBoard, move, noOutputMode) -> nb.types.boolean:
    isPlayerInCheckBeforeMove = chessBoard.IsPlayerInCheck # muss noch geändert werden
    castlingBeforeMove =        chessBoard.Castling.copy()
    enPassantBeforeMove =       chessBoard.EnPassant

    """
    print("move.PieceNum =", move.PieceNum)
    print("move.PiecePos =", move.PiecePos)
    print("move.NewPos =", move.NewPos)
    print("move.MoveID =", move.getMoveID())
    print("move.CapturedPieceNum=", move.CapturedPieceNum)
    print("move.IsCastlingLong  =", move.IsCastlingLong)
    print("move.IsCastlingShort =", move.IsCastlingShort)
    print("move.IsEnpassantMove =", move.IsEnpassantMove)
    """

    chessBoard.playMove(move)
    if __debug__:
        chessBoard.printChessBoard("move")

    chessBoard.setIsPlayerInCheck()
    willPlayerBeInCheck = chessBoard.IsPlayerInCheck

    chessBoard.reverseMove(move, castlingBeforeMove, enPassantBeforeMove, isPlayerInCheckBeforeMove)
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
    ('__MoveID',         nb.int64),
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
        self.__MoveID =         -1
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

        
    def setMoveID(self, moveDict):
        testSet = [self.IsCastlingLong, self.IsCastlingShort, self.IsEnpassantMove]
        if testSet.count(True) > 1:
            print(testSet)
            printError("More than one entry is greater than 1. testSet :", fName = "setMoveID", cName = self.ClassName)
            #print(i for i in testSet)
            raise ValueError("More than one entry is greater than 1")


        startNum = 0
        if not self.IsCastlingLong and not self.IsCastlingShort:
            if self.Piece != self.NewPiece:
                startNum = fn.getPieceNum(self.NewPiece)
        else:
            startNum = fn.getPieceNum("king")
            
        moveString = "".join([str(startNum), str(self.PiecePos[0]), str(self.PiecePos[1]), str(self.NewPos[0]), str(self.NewPos[1])])
        self.__MoveID = moveDict[moveString]
        
    def getMoveID(self):
        if self.__MoveID == -1:
            printError("".join(["self.__MoveID =", str(self.__MoveID)]), fName = "getMoveID", cName = self.ClassName)
            raise ValueError("self.__MoveID is not set correctly")
        return self.__MoveID


############################################################################################################################
#### jited MoveManager class ###############################################################################################
############################################################################################################################
        
MoverSpecs = [
    ('ClassName',       nb.types.string),
    ('Piece',           nb.types.string),
    ('PiecePos',        nb.types.Array(nb.types.int64, 1, 'C')),
    ('CurrentPlayer',   nb.types.string),
    ('CurrentOpponent', nb.types.string),
    ('Deltas',          nb.types.ListType(nb.types.Array(nb.types.int64, 1, 'C'))),
    ('NoOutputMode',    nb.boolean),
]
@jitclass(MoverSpecs)
class MoveManager(object):

    def __init__(self, piece, piecePos, currentPlayer, noOutputMode) -> None:
        self.ClassName = "MoveManager"
        self.Piece = piece
        self.PiecePos = piecePos
        self.CurrentPlayer = currentPlayer
        self.CurrentOpponent = currentPlayer
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
            while validMove == True and capturedPieceNum == 0:
                Del = delta * i #[d*i for d in delta]
                if __debug__:
                    printDebug("".join(["Del = ", makePosString(Del)]), fName = "findNormalMovesFromPiecePos", cName = self.ClassName)
                               
                newPos, validMove, capturedPieceNum = checkMove(self.CurrentPlayer, self.CurrentOpponent, chessBoard, self.Piece, self.PiecePos, Del)
                
                if not validMove:
                    break

                if(self.Piece == "pawn" and self.PiecePos[1] == 6 and newPos[1] == 7): #Pawn promotion
                    pawnPromotionList = ["queen", "rook", "bishop", "knight"]
                    for promotedPiece in pawnPromotionList:
                        moveObject = MoveInformation(self.Piece, promotedPiece, self.PiecePos, newPos, capturedPieceNum)
                        moveObject.setMoveID(moveDict)
                        moveObject.setIsPromotionMove(True)

                        if not willPlayerBeInCheck(chessBoard, moveObject, self.NoOutputMode):
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

        if(chessBoard.Castling[0][0] == 1 and chessBoard.Board[0][1] == 0 and chessBoard.Board[0][2] == 0 and chessBoard.Board[0][3] == 0): #castling long
            if __debug__:
                debugMessage = "".join(["Adding castling LONG to player ", self.CurrentPlayer,"'s castlingMoves."])
                printDebug(debugMessage, fName = "getCastlingMoves", cName = self.ClassName)

            newPos =       np.array([2, 0])
            inBetweenPos = np.array([3, 0])
            moveObject =          MoveInformation(self.Piece, "king", self.PiecePos, newPos,       capturedPieceNum)
            inBetweenMoveObject = MoveInformation(self.Piece, "king", self.PiecePos, inBetweenPos, capturedPieceNum)

            helper_willPlayerBeInCheck = False
            for moveOb in [moveObject, inBetweenMoveObject]:
                moveOb.setIsCastlingLong(True)
                moveOb.setMoveID(moveDict)
                if willPlayerBeInCheck(chessBoard, moveOb, self.NoOutputMode): 
                    helper_willPlayerBeInCheck = True

            if not helper_willPlayerBeInCheck:
                castlingMoves.append(moveObject)

        if(chessBoard.Castling[0][1] == 1 and chessBoard.Board[0][6] == 0 and chessBoard.Board[0][5] == 0): #castling short
            if __debug__:
                debugMessage = "".join(["Adding castling SHORT to player ", self.CurrentPlayer,"'s castlingMoves."])
                printDebug(debugMessage, fName = "getCastlingMoves", cName = self.ClassName)

            newPos =       np.array([6, 0])
            inBetweenPos = np.array([5, 0])
            moveObject =          MoveInformation(self.Piece, "king", self.PiecePos, newPos,       capturedPieceNum)
            inBetweenMoveObject = MoveInformation(self.Piece, "king", self.PiecePos, inBetweenPos, capturedPieceNum)

            helper_willPlayerBeInCheck = False
            for moveOb in [moveObject, inBetweenMoveObject]:
                moveOb.setIsCastlingShort(True)
                moveOb.setMoveID(moveDict)
                if willPlayerBeInCheck(chessBoard, moveOb, self.NoOutputMode): 
                    helper_willPlayerBeInCheck = True

            if not helper_willPlayerBeInCheck:
                castlingMoves.append(moveObject)

        return castlingMoves


    def findEnPassantMoves(self, chessBoard, moveDict) -> list:
        if __debug__:
            debugMessage = " ".join(["En-passant move available for player", self.CurrentPlayer, "'s pawn at", makePosString(self.PiecePos)])
            printDebug(debugMessage, fName = "findEnPassantMoves", cName = self.ClassName)

        enPassantMoves = []
        newPos = np.array([chessBoard.EnPassant, 5])
        capturedPieceNum = 6 #only pawns can be captured via en-passant 
        
        moveObject = MoveInformation(self.Piece, self.Piece, self.PiecePos, newPos, capturedPieceNum) # vielleicht piece = 6 hier?
        moveObject.setIsEnpassantMove(True)
        moveObject.setMoveID(moveDict)

        if not willPlayerBeInCheck(chessBoard, moveObject, self.NoOutputMode):
            enPassantMoves.append(moveObject)

        return enPassantMoves


############################################################################################################################
#### NON-jited MoveDictionary class ########################################################################################
############################################################################################################################
                    
"""
Create the Move dictionary holding all different possible moves in chess (not considering the type of piece that made the move)
"""
SpecsMoveDictionary = [
    ('MoveDict', nb.types.DictType(nb.types.string, nb.int64)),
    ('MoveList', nb.types.ListType(nb.types.string)),
    ('MoveID'  , nb.int64),
]
@jitclass(SpecsMoveDictionary)
class MoveDictionary(object):
    
    def __init__(self) -> None:
        self.MoveDict = nb.typed.Dict.empty(nb.types.string, nb.int64)
        self.MoveList = nb.typed.List(("14020", "14060")) #castling short, castling long
        self.MoveID = 0
        self.initializeMoveDict()

        
    def CPlusPlus(self) -> nb.int64: # C type "i++" function
        self.MoveID += 1
        return self.MoveID - 1

    
    def initializeMoveDict(self) -> None:
        self.MoveDict["14020"] = self.CPlusPlus()  #castling short
        self.MoveDict["14060"] = self.CPlusPlus()  #castling long

        for col in range(8):
            for row in range(8):
                deltas = getBasicPieceMoves("queen", nb.typed.List((row,col)), False)
                self.appendToList("queen", row, col, deltas)
                deltas = getBasicPieceMoves("knight", nb.typed.List((row,col)), False)
                self.appendToList("knight", row, col, deltas)
                deltas = getBasicPieceMoves("pawn", nb.typed.List((row,col)), False)
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
                            moveString = "".join([str(newPieceNumber), str(piecePos[0]), str(piecePos[1]), str(newPos[0]), str(newPos[1])])
                            self.MoveList.append(moveString)
                            self.MoveDict[moveString] = self.CPlusPlus()
                else:
                    #self.MoveList.append((0, piecePos[0], piecePos[1], newPos[0], newPos[1]))
                    #self.MoveDict[(0, piecePos[0], piecePos[1], newPos[0], newPos[1])] = self.CPlusPlus()
                    moveString = "".join(["0", str(piecePos[0]), str(piecePos[1]), str(newPos[0]), str(newPos[1])])
                    self.MoveList.append(moveString)
                    self.MoveDict[moveString] = self.CPlusPlus()
                if(piece == "pawn" or piece == "knight" or piece == "king"):
                    break
                else:
                    i += 1



if __name__ == '__main__': main()
