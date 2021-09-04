import numpy as np
import moves as mv
import functions as fn
from functions import printInfo, printDebug, printError, makePosString
from timeit import default_timer as timer
import pickle
import numba as nb
from numba.experimental import jitclass

############################################################################################################################
#### jited BoardInfo class ########################################################################################## 
############################################################################################################################
BoardInfoSpecs = [
    ('ClassName',       nb.types.string),
    ('Castling',        nb.types.DictType(nb.types.string, nb.boolean)), #nb.types.ListType(nb.types.ListType(nb.boolean))),
    ('KingPositions',   nb.types.DictType(nb.types.string, nb.int64)), #nb.types.Array(nb.types.int64, 2, 'C')),
    ('EnPassant',       nb.types.int64),
    ('CurrentPlayer',   nb.types.string),
    ('CurrentOpponent', nb.types.string),
    ('IsPlayerInCheck', nb.boolean),
    ('Reversed',        nb.boolean),
    ('NoOutputMode',    nb.boolean),
    ('ColoredOutput',   nb.boolean),
]

@jitclass(BoardInfoSpecs)
class BoardInfo(object):

    def __init__(self, noOutputMode, coloredOutput) -> None:
        self.ClassName =       "BoardInfo"
        self.Castling =        nb.typed.Dict.empty(key_type = nb.types.string, value_type = nb.boolean)
        self.KingPositions =   nb.typed.Dict.empty(key_type = nb.types.string, value_type = nb.int64) #nb.types.Array(nb.types.int64, 1, 'C'))
        self.EnPassant =       -99 # Enpassant colum. If -99, en-passant is not allowed in next move
        self.CurrentPlayer =   "white"
        self.CurrentOpponent = "black"
        self.IsPlayerInCheck = False
        self.Reversed  =       False
        self.NoOutputMode =    noOutputMode
        self.ColoredOutput =   coloredOutput
        self.initialize()
        

    def initialize(self) -> None:
        self.KingPositions["white,x"] = 4
        self.KingPositions["white,y"] = 0
        self.KingPositions["black,x"] = 4
        self.KingPositions["black,y"] = 0

        self.Castling["white,long"] =  True
        self.Castling["white,short"] = True
        self.Castling["black,long"] =  True
        self.Castling["black,short"] = True

        
############################################################################################################################
#### jited ChessBoard class ################################################################################################ 
############################################################################################################################
ChessBoardSpecs = [
    ('ClassName', nb.types.string),
    ('Board',     nb.types.Array(nb.types.int64, 2, 'C')),
    ('BoardInfo', BoardInfo.class_type.instance_type),
]

@jitclass(ChessBoardSpecs)
class ChessBoard(object):

    def __init__(self, boardInfo) -> None:
        self.ClassName = "ChessBoard"
        self.Board =     np.zeros((8, 8), dtype = np.int64)
        self.BoardInfo = boardInfo


    def willEnPassantBeAllowedInNextMove(self, move):
        oppSign = fn.getPlayerSign(self.BoardInfo.CurrentOpponent)

        if(move.Piece == "pawn" and move.PiecePos[1] == 1 and move.NewPos[1] == 3 and
           (self.Board[move.NewPos[1]][move.NewPos[0] - 1] == oppSign * 6 or self.Board[move.NewPos[1]][move.NewPos[0] + 1] == oppSign * 6)):
            self.BoardInfo.EnPassant = move.NewPos[0]
        else:
            self.BoardInfo.EnPassant = -99

            
    def simulateMove(self, move) -> None:
        move.checkSpecialMovesSetting()
        
        ##############################################
        #-#-# EnPassant Moves and Castling Moves #-#-#
        ##############################################
        if move.IsEnpassantMove:
            if __debug__:
                printInfo(self.BoardInfo.NoOutputMode, "Player", self.BoardInfo.CurrentPlayer, "just captured via en-passant at", makePosString(move.PiecePos))

            oppPawnPos = np.array([move.NewPos[0], move.PiecePos[1]])
            self.removePiece(oppPawnPos)
        elif move.IsCastlingLong:
            if __debug__:
                printDebug(" ".join(["Player", self.BoardInfo.CurrentPlayer, "is CASTLING long!"]), fName = "playMove", cName = self.ClassName)

            self.removePiece(np.array([0, 0]))
            self.setPiece(self.BoardInfo.CurrentPlayer, "rook", np.array([3, 0]))
        elif move.IsCastlingShort:
            if __debug__:
                printDebug(" ".join(["Player", self.BoardInfo.CurrentPlayer, "is CASTLING short!"]), fName = "playMove", cName = self.ClassName)

            self.removePiece(np.array([7,0]))
            self.setPiece(self.BoardInfo.CurrentPlayer, "rook", np.array([5,0]))
        
        ################################
        #-#-# Play the actual move #-#-#
        ################################
        self.removePiece(move.PiecePos)
        self.setPiece(self.BoardInfo.CurrentPlayer, move.NewPiece, move.NewPos)
        

    def reverseSimulation(self, move, isPlayerInCheckBeforeMove) -> None:
        self.removePiece(move.NewPos)
        self.setPiece(self.BoardInfo.CurrentPlayer, move.Piece, move.PiecePos)

        if move.IsEnpassantMove:
            oppPawn = np.array([move.NewPos[0], move.PiecePos[1]])
            self.setPiece(self.BoardInfo.CurrentOpponent, "pawn", oppPawn)
        elif not move.IsEnpassantMove and move.CapturedPieceNum > 0:
            capturedPiece = fn.getPieceName(move.CapturedPieceNum)
            self.setPiece(self.BoardInfo.CurrentOpponent, capturedPiece, move.NewPos)
        elif move.IsCastlingLong:
            self.removePiece(np.array([3, 0]))
            self.setPiece(self.BoardInfo.CurrentPlayer, "rook", np.array([0, 0]))
        elif move.IsCastlingShort:
            self.removePiece(np.array([5, 0]))
            self.setPiece(self.BoardInfo.CurrentPlayer, "rook", np.array([7,0]))

        self.BoardInfo.IsPlayerInCheck = isPlayerInCheckBeforeMove
        
        
    def setPiece(self, player, piece, pos):
        playerSign = fn.getPlayerSign(player)
        pieceNum = fn.getPieceNum(piece)
        self.Board[pos[1]][pos[0]] = playerSign * pieceNum
        if piece == "king":
            self.BoardInfo.KingPositions[player+",x"] = pos[0]
            self.BoardInfo.KingPositions[player+",y"] = pos[1]

            
    def removePiece(self, pos):
        self.Board[pos[1]][pos[0]] = 0

        
    def setIsPlayerInCheck(self):
        playerIsInCheck = False
        kingPos = np.array([self.BoardInfo.KingPositions[self.BoardInfo.CurrentPlayer+",x"], self.BoardInfo.KingPositions[self.BoardInfo.CurrentPlayer+",y"]])
        ownSign = fn.getPlayerSign(self.BoardInfo.CurrentPlayer)
        oppSign = fn.getPlayerSign(self.BoardInfo.CurrentOpponent)

        pos = np.array([0, 0])
        DeltaMoveList = []
        DeltaMoveList.append(mv.getBasicPieceMoves("bishop", pos, onlyCaptureMoves = False))
        DeltaMoveList.append(mv.getBasicPieceMoves("rook",   pos, onlyCaptureMoves = False))
        DeltaMoveList.append(mv.getBasicPieceMoves("knight", pos, onlyCaptureMoves = False))
        DeltaMoveList.append(mv.getBasicPieceMoves("pawn",   pos, onlyCaptureMoves = True))
        
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

        self.BoardInfo.IsPlayerInCheck = playerIsInCheck

        if self.BoardInfo.IsPlayerInCheck and __debug__:
            printInfo(self.BoardInfo.NoOutputMode, "Player", self.BoardInfo.CurrentPlayer, "is in CHECK!")


    def getPiecesOnBoard(self) -> (nb.typed.List(), nb.typed.List()):
        playerSign = fn.getPlayerSign(self.BoardInfo.CurrentPlayer)
        oppSign =    fn.getPlayerSign(self.BoardInfo.CurrentOpponent)
        playerPieces =   nb.typed.List()
        opponentPieces = nb.typed.List()

        for row in self.Board:
            for col in row:
                if playerSign * col > 0:
                    playerPieces.append(fn.getPieceName(playerSign * col))
                elif oppSign * col > 0:
                    opponentPieces.append(fn.getPieceName(oppSign * col))

        return playerPieces, opponentPieces
            

    def evaluate(self) -> nb.float64:
        playerPieces, opponentPieces = self.getPiecesOnBoard()
        evaluation = 0.0
        
        for piece in playerPieces:
            evaluation += fn.getPieceValue(piece)
        for piece in opponentPieces:
            evaluation -= fn.getPieceValue(piece)

        if evaluation < 0.01:
            evaluation = 0.01

        if __debug__:
            printDebug("".join(["Evaluation = ", evaluation]), fName = "evaluate", cName = "ChessBoard")
            
        return evaluation

    
    def printChessBoard(self, arg = "Board-") -> None:
        lenChessBoard = len(self.Board)
        printInfo(self.BoardInfo.NoOutputMode, "######", arg, "######\n## A B C D E F G H #")
        
        for row in range(lenChessBoard):

            rowInformation = []
            rowInverted = (lenChessBoard -1) - row

            if self.BoardInfo.Reversed == False:
                rowFromPlayerView = rowInverted
            else:
                rowFromPlayerView = row

            rowInformation.append(str(rowInverted + 1))            

            for col in range(lenChessBoard):
                pieceUnicode = fn.getPieceUnicode(self.Board[rowFromPlayerView][col], False)
                rowInformation.append(str(pieceUnicode))

            rowInformation.append(" "+str(rowInverted + 1))

            if not self.BoardInfo.NoOutputMode:
                print(" ".join(rowInformation))

        printInfo(self.BoardInfo.NoOutputMode, "# A B C D E F G H ##\n###### -Board ######")


    def printChessBoardWithInfo(self, moveID = -1, evaluation = 1.0, moveInfo = nb.typed.List(), arg = "Board-", colored = False):
        lenChessBoard = len(self.Board)
        printInfo(self.BoardInfo.NoOutputMode, "######", arg, "######\n## A B C D E F G H #")

        bkgColor = "49"
        printColor = ""
        
        for row in range(lenChessBoard):
            rowInfo = []

            rowInverted = (lenChessBoard -1) - row
            
            if not self.BoardInfo.Reversed:
                rowFromPlayerView = rowInverted
            else:
                rowFromPlayerView = row

            rowInfo.append(str(rowInverted + 1))

            for col in range(lenChessBoard):
                if colored:
                    if   ((8 - row)%2 == 0 and col%2 == 0) or ((8 - row)%2 == 1 and col%2 == 1): bkgColor = "44"
                    elif ((8 - row)%2 == 0 and col%2 == 1) or ((8 - row)%2 == 1 and col%2 == 0): bkgColor = "46"
                                                              
                    if self.Board[rowFromPlayerView][col] < 0:   printColor = "\033[0;30;" + bkgColor + "m"
                    elif self.Board[rowFromPlayerView][col] > 0: printColor = "\033[0;37;" + bkgColor + "m"
                    else:                                        printColor = "\033[1;37;" + bkgColor + "m"

                pieceUnicode = fn.getPieceUnicode(self.Board[rowFromPlayerView][col], colored)
                rowInfo.append(printColor + pieceUnicode)
                
            if colored:
                if row == 2 and len(moveInfo) > 0:
                    rowInfo.append(" ".join(["\033[1;37;49m", str(rowInverted + 1), moveInfo[0], "(\033[1;32;49m" + str(moveID) + "," + fn.floatToString(evaluation) + ")"]))
                elif row == 4 and len(moveInfo) > 1:
                    rowInfo.append(" ".join(["\033[1;37;49m", str(rowInverted + 1), moveInfo[1]]))
                elif row == 6 and len(moveInfo) > 2:
                    rowInfo.append(" ".join(["\033[1;37;49m", str(rowInverted + 1), moveInfo[2]]))
                else:
                    rowInfo.append(" ".join(["\033[1;37;49m", str(rowInverted + 1)]))
            else:
                if row == 2 and len(moveInfo) > 0:
                    rowInfo.append(" ".join(["", str(rowInverted + 1), moveInfo[0], "(" + str(moveID) + "," + fn.floatToString(evaluation) + ")"]))
                elif row == 4 and len(moveInfo) > 1:
                    rowInfo.append(" ".join(["", str(rowInverted + 1), moveInfo[1]]))
                elif row == 4 and len(moveInfo) > 2:
                    rowInfo.append(" ".join(["", str(rowInverted + 1), moveInfo[2]]))
                else:
                    rowInfo.append(" ".join(["", str(rowInverted + 1)]))

            print(" ".join(rowInfo))

        printInfo(self.BoardInfo.NoOutputMode, "# A B C D E F G H ##\n###### -Board ######")



############################################################################################################################
#### jited PieceBoards class ############################################################################################### 
############################################################################################################################

PieceBoardsSpecs = [
    ('ClassName', nb.types.string),
    ('BitBoards', nb.types.Array(nb.types.float64, 4, 'C')),
    ('Pieces',    nb.types.DictType(nb.types.string, nb.types.int64)),
    ('BoardInfo', BoardInfo.class_type.instance_type),
]

@jitclass(PieceBoardsSpecs)
class PieceBoards(object):

    def __init__(self, boardInfo) -> None:
        self.ClassName = "PieceBoards"
        self.BitBoards = np.zeros((2,6,8,8), dtype = np.float64)
        self.BoardInfo = boardInfo
        self.fillPieceDict()


    def fillPieceDict(self):
        self.Pieces = nb.typed.Dict.empty(key_type = nb.types.string, value_type = nb.types.int64)
        self.Pieces["king"]   = 1
        self.Pieces["queen"]  = 2
        self.Pieces["rook"]   = 3
        self.Pieces["bishop"] = 4
        self.Pieces["knight"] = 5
        self.Pieces["pawn"]   = 6        


    def initialize(self):
        self.pawnInitializer()
        self.knightInitializer()
        self.bishopInitializer()
        self.rookInitializer()
        self.queenInitializer()
        self.kingInitializer()
        
        
    def pawnInitializer(self) -> None:
        self.BitBoards[0][5][1] = [1 for i in range(8)] #white
        self.BitBoards[1][5][6] = [1 for i in range(8)] #black

        
    def knightInitializer(self) -> None:
        self.BitBoards[0][4][0][1], self.BitBoards[0][4][0][6] = 1, 1 #white
        self.BitBoards[1][4][7][1], self.BitBoards[1][4][7][6] = 1, 1 #black

        
    def bishopInitializer(self) -> None:
        self.BitBoards[0][3][0][2], self.BitBoards[0][3][0][5] = 1, 1 #white
        self.BitBoards[1][3][7][2], self.BitBoards[1][3][7][5] = 1, 1 #black        

        
    def rookInitializer(self) -> None:
        self.BitBoards[0][2][0][0], self.BitBoards[0][2][0][7] = 1, 1 #white
        self.BitBoards[1][2][7][0], self.BitBoards[1][2][7][7] = 1, 1 #black

        
    def queenInitializer(self) -> None:
        self.BitBoards[0][1][0][3] = 1 #white
        self.BitBoards[1][1][7][3] = 1 #black

        
    def kingInitializer(self) -> None:
        self.BitBoards[0][0][0][4] = 1 #white
        self.BitBoards[1][0][7][4] = 1 #black

        
    def removePiece(self, player, piece, oldPos):
        playerIdx = fn.getPlayerIndex(player)
        pieceIdx = self.Pieces[piece] - 1
        self.BitBoards[playerIdx][pieceIdx][oldPos[1]][oldPos[0]] = 0

        
    def setPiece(self, player, piece, newPos):
        playerIdx = fn.getPlayerIndex(player)
        pieceIdx = self.Pieces[piece] - 1
        self.BitBoards[playerIdx][pieceIdx][newPos[1]][newPos[0]] = 1


############################################################################################################################
#### jited BoardManager class ############################################################################################## 
############################################################################################################################

BoardManagerSpecs = [
    ('ClassName',        nb.types.string),
    ('BoardInfo', BoardInfo.class_type.instance_type),
    ('ChessBoard',       ChessBoard.class_type.instance_type),
    ('PieceBoards',      PieceBoards.class_type.instance_type),
    ('MoveDict',         nb.types.DictType(nb.types.string, nb.int64)),
    ('Players',          nb.types.DictType(nb.types.string, nb.types.int64)),

]

@jitclass(BoardManagerSpecs)
class BoardManager(object):

    def __init__(self, boardInfo, chessBoard, pieceBoards, moveDict) -> None:
        self.ClassName =   "BoardManager"
        self.BoardInfo =   boardInfo
        self.ChessBoard =  chessBoard
        self.PieceBoards = pieceBoards
        self.MoveDict =    moveDict
        self.fillPlayerDict()
        self.initialize()
        
        
    def fillPlayerDict(self):
        self.Players = nb.typed.Dict.empty(key_type = nb.types.string, value_type = nb.types.int64)
        self.Players["white"] = 1
        self.Players["black"] = -1

        
    def initialize(self) -> None:
        self.PieceBoards.initialize()
        self.BoardInfo.initialize()
        self.setPieces()
        
        if __debug__:
            printDebug("ChessBoard and BitBoards initialized:", fName = "initialize", cName = self.ClassName)
            print(self.PieceBoards.BitBoards)


    def setPieces(self) -> None:
        lenChessBoard = len(self.ChessBoard.Board)
        for playerIdx, player in enumerate(self.Players):
            playerSign = self.Players[player]
            for pieceIdx, piece in enumerate(self.PieceBoards.Pieces):
                #printInfo(self.NoOutputMode, "X", str(pieceIdx), str(piece))
                for row in range(lenChessBoard):
                    for col in range(lenChessBoard):
                        if(self.PieceBoards.BitBoards[playerIdx][pieceIdx][row][col] == 1):
                            self.ChessBoard.Board[row][col] = playerSign*(pieceIdx+1)
                            if piece == "king":
                                if player == self.BoardInfo.CurrentPlayer:
                                    self.BoardInfo.KingPositions[player+",x"] = col
                                    self.BoardInfo.KingPositions[player+",y"] = row
                                else:
                                    self.BoardInfo.KingPositions[player+",x"] = col
                                    self.BoardInfo.KingPositions[player+",y"] = lenChessBoard - 1 - row
        if self.BoardInfo.CurrentPlayer == "black":
            self.BoardInfo.Reversed = True
        else:
            self.BoardInfo.Reversed = False
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
                moveManager = mv.MoveManager(piece, piecePos, self.BoardInfo.NoOutputMode)
                moveList = moveManager.findNormalMovesFromPiecePos(self.ChessBoard, self.MoveDict)
                for move in moveList:
                    normalMoves.append(move)

        return normalMoves


    def getCastlingMovesList(self) -> nb.typed.List():
        castlingMoves = nb.typed.List()

        pos = np.array([self.BoardInfo.KingPositions[self.BoardInfo.CurrentPlayer+",x"], self.BoardInfo.KingPositions[self.BoardInfo.CurrentPlayer+",y"]])
        moveManager = mv.MoveManager("king", pos, self.BoardInfo.NoOutputMode)
        moveList = moveManager.findCastlingMoves(self.ChessBoard, self.MoveDict)
        for move in moveList:
            castlingMoves.append(move)

        return castlingMoves

    
    def getEnPassantMovesList(self, pawnPositions) -> nb.typed.List():
        enPassantMoves = nb.typed.List()

        for pawnPos in pawnPositions:
            if pawnPos[1] == 4 and (pawnPos[0] == self.BoardInfo.EnPassant - 1 or pawnPos[0] == self.BoardInfo.EnPassant + 1):
                moveManager = mv.MoveManager("pawn", pawnPos, self.BoardInfo.NoOutputMode)
                moveList = moveManager.findEnPassantMoves(self.ChessBoard, self.MoveDict)
                for move in moveList:
                    enPassantMoves.append(move)

        return enPassantMoves


    def getAllowedMovesList(self):
        ownSign = self.Players[self.BoardInfo.CurrentPlayer]
        oppSign = self.Players[self.BoardInfo.CurrentOpponent]
        ownPositions = fn.getPositions(self.ChessBoard.Board, ownSign)
        oppPositions = fn.getPositions(self.ChessBoard.Board, oppSign)

        if __debug__:
            printDebug(" ".join(["Own piece positions of player:", self.BoardInfo.CurrentPlayer  , ":"]), fName = "getAllowedMovesList", cName = self.ClassName)
            print(ownPositions)
            printDebug(" ".join(["Opp piece positions of player:", self.BoardInfo.CurrentOpponent, ":"]), fName = "getAllowedMovesList", cName = self.ClassName)
            print(oppPositions)

        if not self.BoardInfo.NoOutputMode:
            print("BoardInfo.Castling =",  self.BoardInfo.Castling)
            print("BoardInfo.EnPassant =", self.BoardInfo.EnPassant)

        #-#-# Normal Moves #-#-#
        allMovesList = self.getNormalMovesList(ownPositions)
        if __debug__:
            printDebug("".join(["normalMoves (len = ", str(len(allMovesList)), "):"]), fName = "getAllowedMovesList", cName = self.ClassName)
            for move in allMovesList:
                print("    piece =", move.Piece + "(moveID = ", str(move.getMoveID()) + "):", move.PiecePos + "->" + move.NewPos)

        #-#-# Castling Moves #-#-#
        pos = np.array([self.BoardInfo.KingPositions[self.BoardInfo.CurrentPlayer+",x"], self.BoardInfo.KingPositions[self.BoardInfo.CurrentPlayer+",y"]])
        if(not self.BoardInfo.IsPlayerInCheck and (pos == np.array([4, 0])).all() and
           (self.BoardInfo.Castling[self.BoardInfo.CurrentPlayer+",long"] or
            self.BoardInfo.Castling[self.BoardInfo.CurrentPlayer+",short"])):
            castlingMoves = self.getCastlingMovesList()            
            if __debug__:
                printDebug("castlingMoves :", fName = "getAllowedMovesList", cName = self.ClassName)
                for move in castlingMoves:
                    print("    piece =", move.Piece + "(moveID = ", str(move.getMoveID()) + "):", move.PiecePos + "->" + move.NewPos)
            allMovesList.extend(castlingMoves)

        #-#-# Enpassant Moves #-#-#
        if self.BoardInfo.EnPassant >= 0:
            enPassantMoves = self.getEnPassantMovesList(ownPositions[5])
            if __debug__:
                printDebug("enPassantMoves :", fName = "getAllowedMovesList", cName = self.ClassName)
                for move in enPassantMoves:
                    print("    piece =", move.Piece + "(moveID = ", str(move.getMoveID()) + "):", move.PiecePos + "->" + move.NewPos)
            allMovesList.extend(enPassantMoves)

        return allMovesList
    

    def playMove(self, move, writeMoveInfoList = False) -> nb.typed.List():
        move.checkSpecialMovesSetting()
        returnStringList = nb.typed.List()

        if writeMoveInfoList:
            oldPosReadable = fn.getReadablePosition(self.BoardInfo.CurrentPlayer, move.PiecePos)
            newPosReadable = fn.getReadablePosition(self.BoardInfo.CurrentPlayer, move.NewPos)
            returnStringList.append(" ".join([" ", "Player", self.BoardInfo.CurrentPlayer + ":", move.Piece, "from", oldPosReadable, "to", newPosReadable]))

        #########################
        #-#-# special Moves #-#-#
        #########################
        if move.IsEnpassantMove:
            if __debug__:
                printInfo(self.BoardInfo.NoOutputMode, "Player", self.BoardInfo.CurrentPlayer, "just captured via en-passant at", makePosString(move.PiecePos))
            if writeMoveInfoList:
                returnStringList.append(" ".join([" ", "Player", self.BoardInfo.CurrentPlayer, "captured", fn.getPieceName(move.CapturedPieceNum), "via en-passant"]))
                
            oppPawnPos = np.array([move.NewPos[0], move.PiecePos[1]])
            self.removePiece(self.BoardInfo.CurrentOpponent, "pawn", oppPawnPos)
        elif not move.IsEnpassantMove and move.CapturedPieceNum > 0:
            if __debug__:
                printInfo(self.BoardInfo.NoOutputMode, "Player", self.BoardInfo.CurrentPlayer, "just captured a", fn.getPieceName(move.CapturedPieceNum), "at", makePosString(move.PiecePos))
            if writeMoveInfoList:
                returnStringList.append(" ".join([" ", "Player", self.BoardInfo.CurrentPlayer, "captured", fn.getPieceName(move.CapturedPieceNum), "at", newPosReadable]))

            for pieceOpp in self.PieceBoards.Pieces:
                self.removePiece(self.BoardInfo.CurrentOpponent, pieceOpp, move.NewPos)
        elif move.IsPromotionMove:
            if __debug__:
                printInfo(self.BoardInfo.NoOutputMode, "Player", self.BoardInfo.CurrentPlayer, "just promoted his pawn")
            if writeMoveInfoList:
                returnStringList.append(" ".join([" ", "Player", self.BoardInfo.CurrentPlayer, "'s promoted his pawn to a", move.NewPiece, "at", newPosReadable]))
        elif move.IsCastlingLong:
            if __debug__:
                printDebug(" ".join(["Player", self.BoardInfo.CurrentPlayer, "is CASTLING long!"]), fName = "playMove", cName = self.ClassName)
            if(writeMoveInfoList):
                returnStringList.append(" ".join([" ", "Player", self.BoardInfo.CurrentPlayer, "castled long"]))
                
            self.removePiece(self.BoardInfo.CurrentPlayer, "rook", np.array([0, 0]))
            self.setPiece(   self.BoardInfo.CurrentPlayer, "rook", np.array([3, 0]))
            self.BoardInfo.Castling[self.BoardInfo.CurrentPlayer+",long"] = False
            self.BoardInfo.Castling[self.BoardInfo.CurrentPlayer+",short"] = False
        elif move.IsCastlingShort:
            if __debug__:
                printDebug(" ".join(["Player", self.BoardInfo.CurrentPlayer, "is CASTLING short!"]), fName = "playMove", cName = self.ClassName)
            if(writeMoveInfoList):
                returnStringList.append(" ".join([" ", "Player", self.BoardInfo.CurrentPlayer, "castled short"]))
                
            self.removePiece(self.BoardInfo.CurrentPlayer, "rook", np.array([7, 0]))
            self.setPiece(   self.BoardInfo.CurrentPlayer, "rook", np.array([5, 0]))
            self.BoardInfo.Castling[self.BoardInfo.CurrentPlayer+",long"] = False
            self.BoardInfo.Castling[self.BoardInfo.CurrentPlayer+",short"] = False

        #####################################
        #-#-# other King and Rook Moves #-#-#
        #####################################
        if self.BoardInfo.Castling[self.BoardInfo.CurrentPlayer+",long"] or self.BoardInfo.Castling[self.BoardInfo.CurrentPlayer+",short"]:
            if move.Piece == "king" and (move.PiecePos == np.array([4, 0])).all():
                printInfo(self.BoardInfo.NoOutputMode, "Player", self.BoardInfo.CurrentPlayer, "is no longer allowed to castle")
                self.BoardInfo.Castling[self.BoardInfo.CurrentPlayer+",long"] = False
                self.BoardInfo.Castling[self.BoardInfo.CurrentPlayer+",short"] = False
            elif move.Piece == "rook" and (move.PiecePos == np.array([0, 0])).all():
                if self.BoardInfo.Castling[self.BoardInfo.CurrentPlayer+",long"] == True:
                   printInfo(self.BoardInfo.NoOutputMode, "Player", self.BoardInfo.CurrentPlayer, "is no longer allowed to castle long")
                self.BoardInfo.Castling[self.BoardInfo.CurrentPlayer+",long"] = False
            elif move.Piece == "rook" and (move.PiecePos == np.array([7, 0])).all():
                if self.BoardInfo.Castling[self.BoardInfo.CurrentPlayer+",short"] == True:
                    printInfo(self.BoardInfo.NoOutputMode, "Player", self.BoardInfo.CurrentPlayer, "is no longer allowed to castle short")
                self.BoardInfo.Castling[self.BoardInfo.CurrentPlayer+",short"] = False

        ##################################
        #-#-# Opponent Rook Captures #-#-#
        ##################################    
        if self.BoardInfo.Castling[self.BoardInfo.CurrentOpponent+",long"] and move.CapturedPieceNum == 3 and (move.NewPos == np.array([0, 7])).all():
            self.BoardInfo.Castling[self.BoardInfo.CurrentOpponent+",long"] = False
            if not self.BoardInfo.NoOutputMode:
                print("INFO: Captured Player", self.BoardInfo.CurrentOpponent, "'s rook! Player", self.BoardInfo.CurrentOpponent, "is no longer allowed to castle long.")
                print("self.BoardInfo.Castling =", self.BoardInfo.Castling)
        elif self.BoardInfo.Castling[self.BoardInfo.CurrentOpponent+",short"] and move.CapturedPieceNum == 3 and (move.NewPos == np.array([7, 7])).all():
            self.BoardInfo.Castling[self.BoardInfo.CurrentOpponent+",short"] = False
            if not self.BoardInfo.NoOutputMode:
                printInfo(self.BoardInfo.NoOutputMode, "Captured Player", self.BoardInfo.CurrentOpponent, "'s rook! Player", self.BoardInfo.CurrentOpponent, "is no longer allowed to castle short.")
                print("self.BoardInfo.Castling =", self.BoardInfo.Castling)
                
        ################################
        #-#-# Play the actual move #-#-#
        ################################
        self.removePiece(self.BoardInfo.CurrentPlayer, move.Piece,    move.PiecePos)
        self.setPiece(   self.BoardInfo.CurrentPlayer, move.NewPiece, move.NewPos)
                
        #######################################################
        #-#-# Check if en-passant is allowed in next move #-#-#
        #######################################################
        self.ChessBoard.willEnPassantBeAllowedInNextMove(move)
        
        return returnStringList


    def reverseMove(self, move, castlingBeforeMove, enPassantBeforeMove, isPlayerInCheckBeforeMove):
        self.removePiece(self.BoardInfo.CurrentPlayer, move.NewPiece, move.NewPos)
        self.setPiece(   self.BoardInfo.CurrentPlayer, move.Piece,    move.PiecePos)

        if move.IsEnpassantMove:
            oppPawn = np.array([move.NewPos[0], move.PiecePos[1]])
            self.setPiece(self.BoardInfo.CurrentOpponent, "pawn", oppPawn)
        elif move.CapturedPieceNum != 0 and not move.IsEnpassantMove:
            capturedPiece = fn.getPieceName(move.CapturedPieceNum)
            self.setPiece(self.BoardInfo.CurrentOpponent, capturedPiece, move.NewPos)  
        elif move.IsCastlingLong:
            self.setPiece(   self.BoardInfo.CurrentPlayer, "rook", np.array([0, 0]))
            self.removePiece(self.BoardInfo.CurrentPlayer, "rook", np.array([3, 0]))
        elif move.IsCastlingShort:
            self.setPiece(   self.BoardInfo.CurrentPlayer, "rook", np.array([7,0]))
            self.removePiece(self.BoardInfo.CurrentPlayer, "rook", np.array([5,0]))

        self.BoardInfo.Castling = castlingBeforeMove
        self.BoardInfo.EnPassant = enPassantBeforeMove
        self.BoardInfo.IsPlayerInCheck = isPlayerInCheckBeforeMove

    
    def removePiece(self, player, piece, oldPos):
        self.PieceBoards.removePiece(player, piece, oldPos)
        self.ChessBoard.removePiece(oldPos)

        
    def setPiece(self, player, piece, newPos):
        self.PieceBoards.setPiece(player, piece, newPos)
        self.ChessBoard.setPiece(player, piece, newPos)
        if piece == "king":
            self.BoardInfo.KingPositions[player+"long"] =  newPos[0]
            self.BoardInfo.KingPositions[player+"short"] = newPos[1]

        
    def setPositionFromFile(self, posFile) -> None:
        fileIn = open(posFile,"rb")
        self = pickle.load(fileIn)

        
    def writePositionToFile(self) -> None:
        inputID = self.getInputID(debug)
        fileOut = open("PositionsToPlayFrom/"+inputID+".pkl", "wb")
        pickle.dump([self],fileOut)
        fileOut.close()
