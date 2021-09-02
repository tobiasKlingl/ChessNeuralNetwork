import numpy as np
import moves as mv
import functions as fn
from functions import printInfo, printDebug, printError, makePosString
from timeit import default_timer as timer
import pickle
import numba as nb
from numba.experimental import jitclass


############################################################################################################################
#### jited ChessBoard class ################################################################################################ 
############################################################################################################################

ChessBoardSpecs = [
    ('ClassName',       nb.types.string),
    ('Board',           nb.types.Array(nb.types.int64, 2, 'C')),
    ('Castling',        nb.types.ListType(nb.types.ListType(nb.boolean))),
    ('EnPassant',       nb.types.int64),
    ('CurrentPlayer',   nb.types.string),
    ('CurrentOpponent', nb.types.string),
    ('KingPositions',   nb.types.Array(nb.types.int64, 2, 'C')),
    ('IsPlayerInCheck', nb.boolean),
    ('Reversed',        nb.boolean),
    ('NoOutputMode',    nb.boolean),
    ('ColoredOutput',   nb.boolean),
]

@jitclass(ChessBoardSpecs)
class ChessBoard(object):

    def __init__(self, noOutputMode, coloredOutput) -> None:
        self.ClassName =       "ChessBoard"
        self.Board =           np.zeros((8,8), dtype = np.int64)
        self.Castling  =       nb.typed.List([nb.typed.List([True, True]), nb.typed.List([True, True])])  #np.array([[True, True], [True, True]], dtype = np.boolean)
        self.EnPassant =       -99 # Enpassant colum. If -99, en-passant is not allowed in next move
        self.CurrentPlayer =   "white"
        self.CurrentOpponent = "black"
        self.KingPositions =   np.array([[4,0],[4,0]], dtype = np.int64)
        self.IsPlayerInCheck = False
        self.Reversed  =       False
        self.NoOutputMode =    noOutputMode
        self.ColoredOutput =   coloredOutput


    def playMove(self, move) -> None:
        items = [move.IsCastlingLong, move.IsCastlingShort, move.IsEnpassantMove, move.IsPromotionMove]
        
        if items.count(True) > 1:
            printError(" ".join(["Several of the following items are set to true:"]), fName = "playMove", cName = self.ClassName) 
            print("move.IsCastlingLong =",  move.IsCastlingLong)
            print("move.IsEnpassantMove =", move.IsEnpassantMove)
            print("move.IsCastlingShort =", move.IsCastlingShort)
            print("move.IsPromotionMove =", move.IsPromotionMove)
            raise ValueError("More than one item is true")

        #########################
        #-#-# Capture Moves #-#-#
        #########################
        if move.CapturedPieceNum > 0:
            if move.IsEnpassantMove:
                if __debug__:
                    printInfo(self.NoOutputMode, "Player", self.CurrentPlayer, "just captured via en-passant at", makePosString(move.PiecePos))

                oppPawnPos = np.array([move.NewPos[0], move.NewPos[1] - 1])
                self.removePiece(self.CurrentOpponent, "pawn", oppPawnPos)

            elif not move.IsEnpassantMove:
                for pieceNum in range(1,7):
                    pieceOpp = fn.getPieceName(pieceNum)
                    self.removePiece(self.CurrentOpponent, pieceOpp, move.NewPos)
                    
        self.removePiece(self.CurrentPlayer, move.Piece,         move.PiecePos)
        self.setPiece(   self.CurrentPlayer, move.NewPiece, move.NewPos)

        ##########################
        #-#-# Castling Moves #-#-#
        ##########################
        if move.IsCastlingLong:
            if __debug__:
                printDebug(" ".join(["Player", self.CurrentPlayer, "is CASTLING long!"]), fName = "playMove", cName = self.ClassName)

            self.removePiece(self.CurrentPlayer, "rook", np.array([0, 0]))
            self.setPiece(   self.CurrentPlayer, "rook", np.array([3, 0]))
            self.Castling[0] = nb.typed.List([False, False])

        elif move.IsCastlingShort:
            if __debug__:
                printDebug(" ".join(["Player", self.CurrentPlayer, "is CASTLING short!"]), fName = "playMove", cName = self.ClassName)

            self.removePiece(self.CurrentPlayer, "rook", np.array([7,0]))
            self.setPiece(   self.CurrentPlayer, "rook", np.array([5,0]))
            self.Castling[0] = nb.typed.List([False, False])

        ###############################
        #-#-# King and Rook Moves #-#-#
        ###############################
        if(self.Castling[0][0] == True or self.Castling[0][1] == True):
            if move.Piece == "king" and (move.PiecePos == np.array([4,0]).all()):
                self.Castling[0] = nb.typed.List([False, False])
            elif move.Piece == "rook" and (move.PiecePos == np.array([0, 0])).all():
                self.Castling[0][0] = False
            elif move.Piece == "rook" and (move.PiecePos == np.array([7,0])).all():
                self.Castling[0][1] = False

        ##################################
        #-#-# Opponent Rook Captures #-#-#
        ##################################    
        if self.Castling[1][0] == True and move.CapturedPieceNum == 3 and (move.NewPos == np.array([0,7])).all():
            self.Castling[1][0] = False
        elif self.Castling[1][1] == 1 and move.CapturedPieceNum == 3 and (move.NewPos == np.array([7,7])).all():
            self.Castling[1][1] = False

        oppSign = fn.getPlayerSign(self.CurrentOpponent)
        if(move.Piece == "pawn" and move.PiecePos[1] == 1 and move.NewPos[1] == 3 and
           (self.Board[move.NewPos[1]][move.NewPos[0]-1] == oppSign * 6 or self.Board[move.NewPos[1]][move.NewPos[0] + 1] == oppSign * 6)):
            self.EnPassant = move.NewPos[0]
        else:
            self.EnPassant = -99


    def reverseMove(self, move, castlingBeforeMove, enPassantBeforeMove, isPlayerInCheckBeforeMove) -> None:
        self.removePiece(self.CurrentPlayer, move.NewPiece, move.NewPos)
        self.setPiece(   self.CurrentPlayer, move.Piece,         move.PiecePos)

        if move.CapturedPieceNum != 0 and not move.IsEnpassantMove:
            oppPawn = np.array([move.NewPos[0], move.NewPos[1] - 1])
            self.setPiece(self.CurrentOpponent, "pawn", oppPawn)
        elif move.CapturedPieceNum != 0 and move.IsEnpassantMove:
            capturedPiece = fn.getPieceName(move.CapturedPieceNum)
            self.setPiece(self.CurrentOpponent, capturedPiece, move.NewPos)

        if move.IsCastlingLong:
            self.setPiece(   self.CurrentPlayer, "rook", np.array([0, 0]))
            self.removePiece(self.CurrentPlayer, "rook", np.array([3, 0]))
        elif move.IsCastlingShort:
            self.setPiece(   self.CurrentPlayer, "rook", np.array([7,0]))
            self.removePiece(self.CurrentPlayer, "rook", np.array([5,0]))

        self.Castling = castlingBeforeMove
        self.EnPassant = enPassantBeforeMove
        self.IsPlayerInCheck = isPlayerInCheckBeforeMove

        
    def removePiece(self, player, piece, oldPos):
        self.removePieceChessBoard(oldPos)

        
    def setPiece(self, player, piece, newPos):
        self.setPieceChessBoard(player, piece, newPos)

        
    def removePieceChessBoard(self, oldPos):
        self.Board[oldPos[1]][oldPos[0]] = 0

        
    def setPieceChessBoard(self, player, piece, newPos):
        playerSign = fn.getPlayerSign(player)
        pieceNum = fn.getPieceNum(piece)
        self.Board[newPos[1]][newPos[0]] = playerSign * pieceNum
            

    def setIsPlayerInCheck(self):
        playerIsInCheck = False
        kingPos = self.KingPositions[0]
        ownSign = fn.getPlayerSign(self.CurrentPlayer)
        oppSign = fn.getPlayerSign(self.CurrentOpponent)

        pos = np.array([0, 0])
        DeltaMoveList = []
        DeltaMoveList.extend(mv.getBasicPieceMoves("bishop", pos, onlyCaptureMoves = False))
        DeltaMoveList.extend(mv.getBasicPieceMoves("rook",   pos, onlyCaptureMoves = False))
        DeltaMoveList.extend(mv.getBasicPieceMoves("knight", pos, onlyCaptureMoves = False))
        DeltaMoveList.extend(mv.getBasicPieceMoves("pawn",   pos, onlyCaptureMoves = True))
        
        deltaMoveList = nb.typed.List(DeltaMoveList)

        for num, deltas in enumerate(deltaMoveList):
            for delta in deltas:
                i = 1
                while i <= 8:
                    pos = kingPos + delta * i

                    if pos[0] < 0 or pos[0] > 7 or pos[1] < 0 or pos[1] > 7 or (ownSign * self.Board[pos[1]][pos[0]] > 0):
                        break
                    elif oppSign * self.Board[pos[1]][pos[0]] > 0:
                        oppK, oppQ, oppR, oppB, oppN, oppP = oppSign * 1, oppSign * 2, oppSign * 3, oppSign * 4, oppSign * 5, oppSign * 6
                        if( (num == 0 and (self.Board[pos[1]][pos[0]] == oppQ or self.Board[pos[1]][pos[0]] == oppB or (i == 1 and self.Board[pos[1]][pos[0]] == oppK))) or
                            (num == 1 and (self.Board[pos[1]][pos[0]] == oppQ or self.Board[pos[1]][pos[0]] == oppR or (i == 1 and self.Board[pos[1]][pos[0]] == oppK))) or
                            (num == 2 and  self.Board[pos[1]][pos[0]] == oppN) or
                            (num == 3 and  self.Board[pos[1]][pos[0]] == oppP) ):
                            playerIsInCheck = True
                        elif num == 0 or num == 1: # macht das überhaupt was??
                            break
                    if num == 2 or num == 3:
                        break
                    i += 1

        self.IsPlayerInCheck = playerIsInCheck

        if self.IsPlayerInCheck:
            printInfo(self.NoOutputMode, "Player", self.CurrentPlayer, "is in CHECK!")


    def printChessBoard(self, arg = "Board-") -> None:
        lenChessBoard = len(self.Board)
        printInfo(self.NoOutputMode, "######", arg, "######\n## A B C D E F G H #")
        
        for row in range(lenChessBoard):
            rowInformation = []

            if self.Reversed == False:
                row = (lenChessBoard -1) - row
            rowInformation.append(str(row + 1))            

            for col in range(lenChessBoard):
                pieceUnicode = fn.getPieceUnicode(self.Board[row][col], False)
                rowInformation.append(str(pieceUnicode))
            rowInformation.append(" "+str(row + 1))

            if not self.NoOutputMode:
                print(" ".join(rowInformation))

        printInfo(self.NoOutputMode, "# A B C D E F G H ##\n###### -Board ######")


    def printChessBoardWithInfo(self, moveID = -1, evaluation = 1, moveInfo = nb.typed.List(), arg = "Board-", colored = False):
        lenChessBoard = len(self.Board)
        printInfo(self.NoOutputMode, "######", arg, "######\n## A B C D E F G H #")

        bkgColor = "49"
        
        for row in range(lenChessBoard):
            rowInfo = []

            if not self.Reversed:
                row = (lenChessBoard -1) - row

            rowInfo.append(str(row + 1))            

            for col in range(lenChessBoard):
                if colored:
                    if   ((8 - row)%2 == 0 and col%2 == 0) or ((8 - row)%2 == 1 and col%2 == 1): bkgColor = "44"
                    elif ((8 - row)%2 == 0 and col%2 == 1) or ((8 - row)%2 == 1 and col%2 == 0): bkgColor = "46"
                                                              
                    if self.Board[row][col] < 0:   printColor = "\033[0;30;" + bkgColor + "m"
                    elif self.Board[row][col] > 0: printColor = "\033[0;37;" + bkgColor + "m"
                    else:                          printColor = "\033[1;37;" + bkgColor + "m"

                pieceUnicode = fn.getPieceUnicode(self.Board[row][col], colored)
                rowInfo.append(printColor + pieceUnicode)
                
            if colored:
                if row == 6 and len(moveInfo) > 0:
                    rowInfo.append(" ".join(["\033[1;37;49m", str(row + 1), moveInfo[0], "(\033[1;32;49m" + str(moveID) + "," + str(evaluation) + ")"]))
                elif row == 4 and len(moveInfo) > 1:
                    rowInfo.append(" ".join(["\033[1;37;49m", str(row + 1), moveInfo[1]]))
                elif row == 2 and len(moveInfo) > 2:
                    rowInfo.append(" ".join(["\033[1;37;49m", str(row + 1), moveInfo[2]]))
                else:
                    rowInfo.append(" ".join(["\033[1;37;49m", str(row + 1)]))
            else:
                if row == 6 and len(moveInfo) > 0:
                    rowInfo.append(" ".join(["", str(row + 1), moveInfo[0], "(" + str(moveID) + "," + str(evaluation) + ")"]))
                elif row == 4 and len(moveInfo) > 1:
                    rowInfo.append(" ".join(["", str(row + 1), moveInfo[1]]))
                elif row == 4 and len(moveInfo) > 2:
                    rowInfo.append(" ".join(["", str(row + 1), moveInfo[2]]))
                else:
                    rowInfo.append(" ".join(["", str(row + 1)]))

            print(" ".join(rowInfo))

        printInfo(self.NoOutputMode, "# A B C D E F G H ##\n###### -Board ######")


############################################################################################################################
#### jited PieceBoards class ############################################################################################### 
############################################################################################################################

PieceBoardsSpecs = [
    ('ClassName',   nb.types.string),
    ('PieceBoards', nb.types.Array(nb.types.float64, 4, 'C')), #nb.int64[:,:,:,:]),
    ('Pieces',          nb.types.DictType(nb.types.string, nb.types.int64)), #nb.types.ListType(nb.types.string)),
    ('NoOutputMode',    nb.boolean),
]

@jitclass(PieceBoardsSpecs)
class PieceBoards(object):

    def __init__(self, noOutputMode) -> None:
        self.ClassName =    "PieceBoards"
        self.PieceBoards =  np.zeros((2,6,8,8), dtype = np.float64)
        self.fillPieceDict()
        self.NoOutputMode = noOutputMode


    def fillPieceDict(self):
        self.Pieces = nb.typed.Dict.empty(key_type = nb.types.string, value_type = nb.types.int64)
        self.Pieces["king"]   = 1
        self.Pieces["queen"]  = 2
        self.Pieces["rook"]   = 3
        self.Pieces["bishop"] = 4
        self.Pieces["knight"] = 5
        self.Pieces["pawn"]   = 6        

        
    def pawnInitializer(self) -> None:
        self.PieceBoards[0][5][1] = [1 for i in range(8)] #white
        self.PieceBoards[1][5][6] = [1 for i in range(8)] #black

        
    def knightInitializer(self) -> None:
        self.PieceBoards[0][4][0][1], self.PieceBoards[0][4][0][6] = 1, 1 #white
        self.PieceBoards[1][4][7][1], self.PieceBoards[1][4][7][6] = 1, 1 #black

        
    def bishopInitializer(self) -> None:
        self.PieceBoards[0][3][0][2], self.PieceBoards[0][3][0][5] = 1, 1 #white
        self.PieceBoards[1][3][7][2], self.PieceBoards[1][3][7][5] = 1, 1 #black        

        
    def rookInitializer(self) -> None:
        self.PieceBoards[0][2][0][0], self.PieceBoards[0][2][0][7] = 1, 1 #white
        self.PieceBoards[1][2][7][0], self.PieceBoards[1][2][7][7] = 1, 1 #black

        
    def queenInitializer(self) -> None:
        self.PieceBoards[0][1][0][3] = 1 #white
        self.PieceBoards[1][1][7][3] = 1 #black

        
    def kingInitializer(self) -> None:
        self.PieceBoards[0][0][0][4] = 1 #white
        self.PieceBoards[1][0][7][4] = 1 #black

        
    def removePiecePieceBoards(self, player, piece, oldPos):
        playerIdx = fn.getPlayerIndex(player)
        pieceIdx = self.Pieces[piece] - 1
        self.PieceBoards[playerIdx][pieceIdx][oldPos[1]][oldPos[0]] = 0

        
    def setPieces(self, player, piece, newPos):
        playerIdx = fn.getPlayerIndex(player)
        pieceIdx = self.Pieces[piece] - 1
        self.PieceBoards[playerIdx][pieceIdx][newPos[1]][newPos[0]] = 1
        

############################################################################################################################
#### jited BoardManager class ############################################################################################## 
############################################################################################################################

BoardManagerSpecs = [
    ('ClassName',       nb.types.string),
    ('ChessBoard',      ChessBoard.class_type.instance_type),
    ('PieceBoards',     PieceBoards.class_type.instance_type),
    ('MoveDict',        nb.types.DictType(nb.types.string, nb.int64)),
    ('Players',         nb.types.DictType(nb.types.string, nb.types.int64)),
    ('CurrentPlayer',   nb.types.string),
    ('CurrentOpponent', nb.types.string),
    ('NoOutputMode',    nb.boolean),
    ('ColoredOutput',   nb.boolean),
]

@jitclass(BoardManagerSpecs)
class BoardManager(object):

    def __init__(self, chessBoard, pieceBoards, moveDict, noOutputMode, coloredOutput) -> None:
        self.ClassName =      "BoardManager"
        self.ChessBoard =      chessBoard
        self.PieceBoards =     pieceBoards
        self.MoveDict =        moveDict
        self.fillPlayerDict()
        self.CurrentPlayer =   "white"
        self.CurrentOpponent = "black"
        self.NoOutputMode =    noOutputMode
        self.ColoredOutput =   coloredOutput

        
    def fillPlayerDict(self):
        self.Players = nb.typed.Dict.empty(key_type = nb.types.string, value_type = nb.types.int64)
        self.Players["white"] = 1
        self.Players["black"] = -1

        
    def initializePieceBoards(self) -> None:
        self.PieceBoards.pawnInitializer()
        self.PieceBoards.knightInitializer()
        self.PieceBoards.bishopInitializer()
        self.PieceBoards.rookInitializer()
        self.PieceBoards.queenInitializer()
        self.PieceBoards.kingInitializer()

        self.ChessBoard.KingPositions = np.array([[4,0],[4,0]], dtype = np.int64)
        self.ChessBoard.Castling = nb.typed.List([nb.typed.List([True, True]), nb.typed.List([True, True])]) # [[white long,white short],[black long, black short]] still allowed?

        if __debug__:
            printDebug("All PieceBoards initialized:", fName = "initializePieceBoards", cName = self.ClassName)
            print(self.PieceBoards.PieceBoards)

            
    def setPieces(self) -> None:
        lenChessBoard = len(self.ChessBoard.Board)
        for playerIdx, player in enumerate(self.Players):
            playerSign = self.Players[player]
            for pieceIdx, piece in enumerate(self.PieceBoards.Pieces):
                #printInfo(self.NoOutputMode, "X", str(pieceIdx), str(piece))
                for row in range(lenChessBoard):
                    for col in range(lenChessBoard):
                        if(self.PieceBoards.PieceBoards[playerIdx][pieceIdx][row][col] == 1):
                            self.ChessBoard.Board[row][col] = playerSign*(pieceIdx+1)
                            if piece == "king":
                                if player == self.CurrentPlayer:
                                    self.ChessBoard.KingPositions[0] = [col, row]
                                else:
                                    self.ChessBoard.KingPositions[1] = [col, (lenChessBoard - 1) - row]
        if self.CurrentPlayer == "black":
            self.ChessBoard.Reversed = True
        else:
            self.ChessBoard.Reversed = False
        if __debug__:
            printDebug("All pieces placed onto ChessBoard", fName = "setPieces", cName = self.ClassName)

        self.ChessBoard.printChessBoard()


    def getNormalMovesList(self, ownPositions) -> nb.typed.List():
        normalMoves = nb.typed.List()
        onlyCaptureMoves = False

        for pieceIdx, piece in enumerate(self.PieceBoards.Pieces):
            if __debug__:
                printDebug(" ".join(["Current piece =", piece + "; ownPositions:"]), fName = "getNormalMoves", cName = self.ClassName)
                print(ownPositions[pieceIdx])

            for piecePos in ownPositions[pieceIdx]:
                moveManager = mv.MoveManager(piece, piecePos, self.CurrentPlayer, self.NoOutputMode)
                moveList = moveManager.findNormalMovesFromPiecePos(self.ChessBoard, self.MoveDict)
                for move in moveList:
                    normalMoves.append(move)

        return normalMoves


    def getCastlingMovesList(self) -> nb.typed.List():
        castlingMoves = nb.typed.List()

        if self.ChessBoard.Castling[0][0] == True or self.ChessBoard.Castling[0][1] == True:
            moveManager = mv.MoveManager("king", self.ChessBoard.KingPositions[0], self.CurrentPlayer, self.NoOutputMode)
            moveList = moveManager.findCastlingMoves(self.ChessBoard, self.MoveDict)
            for move in moveList:
                castlingMoves.append(move)

        return castlingMoves

    
    def getEnPassantMovesList(self, pawnPositions) -> nb.typed.List():
        enPassantMoves = nb.typed.List()

        for pawnPos in pawnPositions:
            if pawnPos[1] == 4 and (pawnPos[0] == self.ChessBoard.EnPassant - 1 or pawnPos[0] == self.ChessBoard.EnPassant + 1):
                moveManager = mv.MoveManager("pawn", pawnPos, self.CurrentPlayer, self.NoOutputMode)
                moveList = moveManager.findEnPassantMoves(self.ChessBoard, self.MoveDict)
                for move in moveList:
                    enPassantMoves.append(move)

        return enPassantMoves


    def getAllowedMovesList(self):
        ownSign = self.Players[self.CurrentPlayer]
        oppSign = self.Players[self.CurrentOpponent]
        ownPositions = fn.getPositions(self.ChessBoard.Board, ownSign)
        oppPositions = fn.getPositions(self.ChessBoard.Board, oppSign)

        if __debug__:
            printDebug(" ".join(["Own piece positions of player:", self.CurrentPlayer  , ":"]), fName = "getAllowedMovesList", cName = self.ClassName)
            print(ownPositions)
            printDebug(" ".join(["Opp piece positions of player:", self.CurrentOpponent, ":"]), fName = "getAllowedMovesList", cName = self.ClassName)
            print(oppPositions)

        if not self.NoOutputMode:
            print("self.ChessBoard.Castling =", self.ChessBoard.Castling)
            print("self.ChessBoard.EnPassant=", self.ChessBoard.EnPassant)

        #-#-# Normal Moves #-#-#
        allMovesList = self.getNormalMovesList(ownPositions)
        if __debug__:
            printDebug("".join(["normalMoves (len = ", str(len(allMovesList)), "):"]), fName = "getAllowedMovesList", cName = self.ClassName)
            for move in allMovesList:
                print("    piece =", move.Piece + "(moveID = ", str(move.getMoveID()) + "):", move.PiecePos, "->", move.NewPos)

        #-#-# Castling Moves #-#-#
        if self.ChessBoard.IsPlayerInCheck == False and (self.ChessBoard.Castling[0][0] == True or self.ChessBoard.Castling[0][1] == True):
            castlingMoves = self.getCastlingMovesList()            
            if __debug__:
                printDebug("castlingMoves :", fName = "getAllowedMovesList", cName = self.ClassName)
                for move in castlingMoves:
                    print("    piece =", move.Piece + "(moveID = ", str(move.getMoveID()) + "):", move.PiecePos, "->", move.NewPos)
            allMovesList.extend(castlingMoves)
            #for move in castlingMoves:
            #    allMovesList.append(move)

        #-#-# Enpassant Moves #-#-#
        if self.ChessBoard.EnPassant >= 0:
            enPassantMoves = self.getEnPassantMovesList(ownPositions[5])
            if __debug__:
                printDebug("enPassantMoves :", fName = "getAllowedMovesList", cName = self.ClassName)
                for move in enPassantMoves:
                    print("    piece =", move.Piece + "(moveID = ", str(move.getMoveID()) + "):", move.PiecePos, "->", move.NewPos)
            for move in enPassantMoves:
                allMovesList.append(move)

        if len(allMovesList) > 0:
            fn.moveProbs(allMovesList)

        return allMovesList
    

    def playMove(self, move, writeMoveInfoList = False) -> nb.typed.List():
        returnStringList = nb.typed.List()
        
        items = [move.IsCastlingLong, move.IsCastlingShort, move.IsEnpassantMove, move.IsPromotionMove]
        
        if items.count(True) > 1:
            printError(" ".join(["Several of the following items are set to true:"]), fName = "playMove", cName = self.ClassName) 
            print("move.IsCastlingLong =",  move.IsCastlingLong)
            print("move.IsEnpassantMove =", move.IsEnpassantMove)
            print("move.IsCastlingShort =", move.IsCastlingShort)
            print("move.IsPromotionMove =", move.IsPromotionMove)
            raise ValueError("More than one item is true")

        if writeMoveInfoList:
            oldPosReadable = fn.getReadablePosition(self.CurrentPlayer, move.PiecePos)
            newPosReadable = fn.getReadablePosition(self.CurrentPlayer, move.NewPos)
            returnStringList.append(" ".join(["Player ", self.CurrentPlayer + ":", move.Piece, "from", oldPosReadable, "to", newPosReadable]))

        #########################
        #-#-# Capture Moves #-#-#
        #########################
        if move.CapturedPieceNum > 0:
            if move.IsEnpassantMove:
                if __debug__:
                    printInfo(self.NoOutputMode, "Player", self.CurrentPlayer, "just captured via en-passant at", makePosString(move.PiecePos))

                oppPawnPos = np.array([move.NewPos[0], move.NewPos[1] - 1])
                self.removePiece(self.CurrentOpponent, "pawn", oppPawnPos)

                if writeMoveInfoList:
                    returnStringList.append(" ".join(["Player", self.CurrentPlayer, "captured", fn.getPieceName(move.CapturedPieceNum), "via en-passant"]))
            elif not move.IsEnpassantMove:
                if writeMoveInfoList:
                    returnStringList.append(" ".join(["Player", self.CurrentPlayer, "captured", fn.getPieceName(move.CapturedPieceNum), "at", newPosReadable]))

                for pieceOpp in self.PieceBoards.Pieces:
                    self.removePiece(self.CurrentOpponent, pieceOpp, move.NewPos)
                    

        self.removePiece(self.CurrentPlayer, move.Piece,    move.PiecePos)
        self.setPiece(   self.CurrentPlayer, move.NewPiece, move.NewPos)

        ###########################
        #-#-# Promotion Moves #-#-#
        ###########################
        if move.IsPromotionMove and writeMoveInfoList:
            returnStringList.append(" ".join(["Player", self.CurrentPlayer, "'s promoted his pawn to a", move.NewPiece, "at", newPosReadable]))

        ##########################
        #-#-# Castling Moves #-#-#
        ##########################
        if move.IsCastlingLong:
            if __debug__:
                printDebug(" ".join(["Player", self.CurrentPlayer, "is CASTLING long!"]), fName = "playMove", cName = self.ClassName)

            self.removePiece(self.CurrentPlayer, "rook", np.array([0, 0]))
            self.setPiece(   self.CurrentPlayer, "rook", np.array([3, 0]))
            self.ChessBoard.Castling[0] = nb.typed.List([False, False])

            if(writeMoveInfoList):
                returnStringList.append(" ".join(["Player", self.CurrentPlayer, "castled long"]))
        elif move.IsCastlingShort:
            if __debug__:
                printDebug(" ".join(["Player", self.CurrentPlayer, "is CASTLING short!"]), fName = "playMove", cName = self.ClassName)

            self.removePiece(self.CurrentPlayer, "rook", np.array([7,0]))
            self.setPiece(   self.CurrentPlayer, "rook", np.array([5,0]))
            self.ChessBoard.Castling[0] = nb.typed.List([False, False])

            if(writeMoveInfoList):
                returnStringList.append(" ".join(["Player", self.CurrentPlayer, "castled short"]))

        ###############################
        #-#-# King and Rook Moves #-#-#
        ###############################
        if(self.ChessBoard.Castling[0][0] == True or self.ChessBoard.Castling[0][1] == True):
            if move.Piece == "king" and (move.PiecePos == np.array([4,0]).all()):
                printInfo(self.NoOutputMode, "Player", self.CurrentPlayer, "is no longer allowed to castle")
                self.ChessBoard.Castling[0] = nb.typed.List([False, False])
            elif move.Piece == "rook" and (move.PiecePos == np.array([0, 0])).all():
                if self.ChessBoard.Castling[0][0] == True:
                   printInfo(self.NoOutputMode, "Player", self.CurrentPlayer, "is no longer allowed to castle long")
                self.ChessBoard.Castling[0][0] = False
            elif move.Piece == "rook" and (move.PiecePos == np.array([7,0])).all():
                if self.ChessBoard.Castling[0][1] == True:
                    printInfo(self.NoOutputMode, "Player", self.CurrentPlayer, "is no longer allowed to castle short")
                self.ChessBoard.Castling[0][1] = False

        ##################################
        #-#-# Opponent Rook Captures #-#-#
        ##################################    
        if self.ChessBoard.Castling[1][0] == True and move.CapturedPieceNum == 3 and (move.NewPos == np.array([0,7])).all():
            self.ChessBoard.Castling[1][0] = False
            if not self.NoOutputMode:
                print("INFO: Captured Player", self.CurrentOpponent, "'s rook! Player", self.CurrentOpponent, "is no longer allowed to castle long.")
                print("self.ChessBoard.Castling=", self.ChessBoard.Castling)
        elif self.ChessBoard.Castling[1][1] == 1 and move.CapturedPieceNum == 3 and (move.NewPos == np.array([7,7])).all():
            self.ChessBoard.Castling[1][1] = False
            if not self.NoOutputMode:
                printInfo(self.NoOutputMode, "Captured Player", self.CurrentOpponent, "'s rook! Player", self.CurrentOpponent, "is no longer allowed to castle short.")
                print("self.ChessBoard.Castling =", self.ChessBoard.Castling)

        oppSign = self.Players[self.CurrentOpponent]
        if(move.Piece == "pawn" and move.PiecePos[1] == 1 and move.NewPos[1] == 3 and
           (self.ChessBoard.Board[move.NewPos[1]][move.NewPos[0]-1] == oppSign * 6 or self.ChessBoard.Board[move.NewPos[1]][move.NewPos[0] + 1] == oppSign * 6)):
            self.ChessBoard.EnPassant = move.NewPos[0]
        else:
            self.ChessBoard.EnPassant = -99
        return returnStringList


    def reverseMove(self, move, castlingBeforeMove, enPassantBeforeMove, isPlayerInCheckBeforeMove):
        self.removePiece(self.CurrentPlayer, move.NewPiece, move.NewPos)
        self.setPiece(   self.CurrentPlayer, move.Piece,    move.PiecePos)

        if move.CapturedPieceNum != 0 and not move.IsEnpassantMove:
            oppPawn = np.array([move.NewPos[0], move.NewPos[1] - 1])
            self.setPiece(self.CurrentOpponent, "pawn", oppPawn)
        elif move.CapturedPieceNum != 0 and move.IsEnpassantMove:
            capturedPiece = fn.getPieceName(move.CapturedPieceNum)
            self.setPiece(self.CurrentOpponent, capturedPiece, move.NewPos)

        if move.IsCastlingLong:
            self.setPiece(   self.CurrentPlayer, "rook", np.array([0, 0]))
            self.removePiece(self.CurrentPlayer, "rook", np.array([3, 0]))
        elif move.IsCastlingShort:
            self.setPiece(   self.CurrentPlayer, "rook", np.array([7,0]))
            self.removePiece(self.CurrentPlayer, "rook", np.array([5,0]))

        self.ChessBoard.Castling = castlingBeforeMove
        self.ChessBoard.EnPassant = enPassantBeforeMove
        self.ChessBoard.IsPlayerInCheck = isPlayerInCheckBeforeMove

    
    def removePiece(self, player, piece, oldPos):
        self.PieceBoards.removePiecePieceBoards(player, piece, oldPos)
        self.ChessBoard.removePieceChessBoard(oldPos)

        
    def setPiece(self, player, piece, newPos):
        self.PieceBoards.setPieces(player, piece, newPos)
        self.ChessBoard.setPieceChessBoard(player, piece, newPos)
        if(piece == "king"):
            self.ChessBoard.KingPositions[0] = newPos

        
    def setPositionFromFile(self, posFile) -> None:
        fileIn = open(posFile,"rb")
        self = pickle.load(fileIn)

        
    def writePositionToFile(self) -> None:
        inputID = self.getInputID(debug)
        fileOut = open("PositionsToPlayFrom/"+inputID+".pkl", "wb")
        pickle.dump([self],fileOut)
        fileOut.close()
