from Board import BoardManager
import functions as fn
from functions import printInfo, printDebug, printError, makePosString
import numpy as np
import numba as nb
from numba.experimental import jitclass
import random

############################################################################################################################
#### jited GameManager class ############################################################################################## 
############################################################################################################################

GameManagerSpecs = [
    ('ClassName',     nb.types.string),
    ('GameMode',      nb.types.string),
    ('BoardManager',  BoardManager.class_type.instance_type),
    ('PlyNumber',     nb.int64),
    ('Finished',      nb.boolean),
    ('Winner',        nb.types.string),
    ('NoOutputMode',  nb.boolean),
    ('ColoredOutput', nb.boolean),
]

@jitclass(GameManagerSpecs)
class GameManager(object):

    def __init__(self, boardManager, noOutputMode, coloredOutput) -> None:
        self.ClassName =     "GameManager"
        self.GameMode =      "SelfplayRandomVsRandom"
        self.BoardManager =  boardManager
        self.PlyNumber =     1
        self.Finished =      False
        self.Winner =        ""
        self.NoOutputMode =  noOutputMode
        self.ColoredOutput = coloredOutput


    def getPiecesOnBoard(self) -> (nb.typed.List(), nb.typed.List()):
        printInfo(self.NoOutputMode, "Less than 9 moves available! Check if game is decided.")
        
        playerSign = self.BoardManager.Players[self.BoardManager.CurrentPlayer]
        oppSign =    self.BoardManager.Players[self.BoardManager.CurrentOpponent]
        playerPieces =   nb.typed.List()
        opponentPieces = nb.typed.List()

        for row in self.BoardManager.ChessBoard.Board:
            for col in row:
                if playerSign * col > 0:
                    playerPieces.append(fn.getPieceName(playerSign * col))
                elif oppSign * col > 0:
                    opponentPieces.append(fn.getPieceName(oppSign * col))

        if not self.NoOutputMode:
            print("playerPieces = ", playerPieces)
            print("opponentPieces = ", opponentPieces)
            
        return playerPieces, opponentPieces


    def getMoveNumToPlay(self, allMoves) -> nb.int64:
        numMoves = len(allMoves)

        if self.GameMode == "SelfplayRandomVsRandom" or (self.GameMode == "SelfplayNetworkVsRandom" and self.BoardManager.CurrentPlayer == "black"):
            printInfo(self.NoOutputMode, "Random mover is moving now!")
            n_move = random.randint(0, numMoves - 1)
        elif self.GameMode == "SelfplayNetworkVsNetwork" or (self.GameMode == "SelfplayNetworkVsRandom" and self.BoardManager.CurrentPlayer == "white"):
            printInfo(self.NoOutputMode, "Neural network is moving now!")
            n_move = fn.rand_choice_nb(np.arange(numMoves), allMoves)

        printInfo(self.NoOutputMode, " ".join(["numMoves =", str(numMoves), "; Move number n_move =", str(n_move), "chosen"]))
            
        return n_move

    
    def findNextMove(self) -> (nb.int64, nb.float64):
        self.BoardManager.ChessBoard.setIsPlayerInCheck()
        
        self.Finished = False
        moveID = -1
        moveEval = -1

        allMoves = self.BoardManager.getAllowedMovesList()
        numMoves = len(allMoves)
        print("lenAllMoves =", numMoves)
        
        if numMoves == 0: #No more moves available 0
            if self.BoardManager.ChessBoard.IsPlayerInCheck:
                printInfo(self.NoOutputMode, "Player", self.BoardManager.CurrentPlayer, "is CHECKMATE!")
                self.Winner = self.BoardManager.CurrentOpponent
            elif not self.BoardManager.ChessBoard.IsPlayerInCheck:
                printInfo(self.NoOutputMode, "Player", self.BoardManager.CurrentPlayer, "has no more moves available => Remis.")
                self.Winner = "draw"

            self.Finished = True
        elif numMoves > 0:
            if numMoves <= 8:
                #Check if only king is left and opponent can still win!
                playerPieces, opponentPieces = self.getPiecesOnBoard()
                
                if len(playerPieces) == 1:
                    if len(opponentPieces) > 2 or len(opponentPieces) > 1 and ("rook" in opponentPieces or "queen" in opponentPieces):
                        for rand, move in enumerate(allMoves):
                            if move.CapturedPieceNum > 0:
                                moveInfoList = self.BoardManager.playMove(move, writeMoveInfoList = True)
                                if not self.NoOutputMode:
                                    self.BoardManager.ChessBoard.printChessBoardWithInfo(moveID = move.getMoveID(), evaluation = 1.0, moveInfo = moveInfoList, colored = self.ColoredOutput)
                                self.Finished = False
                                return -1, 1

                        self.Winner = self.BoardManager.CurrentOpponent
                        self.Finished = True
                        return -1, 1
                    elif len(opponentPieces) == 2 and ("bishop" in opponentPieces or "knight" in opponentPieces):
                        self.Winner = "draw"
                        self.Finished = True
                        return -1, 1

            if not self.NoOutputMode:
                fn.printMoves(self.BoardManager.CurrentPlayer, allMoves, self.ColoredOutput, self.NoOutputMode)
            
            n_move = self.getMoveNumToPlay(allMoves)
            move = allMoves[n_move]
            moveID = move.getMoveID()
            moveEval = move.MoveEvaluation
            
            moveInfoList = self.BoardManager.playMove(move, True)

            if not self.NoOutputMode:
                self.BoardManager.ChessBoard.printChessBoardWithInfo(moveID = moveID, evaluation = moveEval, moveInfo = moveInfoList, colored = self.ColoredOutput)
                print("self.BoardManager.ChessBoard.Castling/ EnPassant =",  self.BoardManager.ChessBoard.Castling, "/", self.BoardManager.ChessBoard.EnPassant)
                
        return moveID, moveEval


    def nextPly(self):
        # Reverse the board
        if self.BoardManager.ChessBoard.Reversed:
            self.BoardManager.ChessBoard.Reversed = False
        elif not self.BoardManager.ChessBoard.Reversed:
            self.BoardManager.ChessBoard.Reversed = True

        self.BoardManager.ChessBoard.Board[:] = self.BoardManager.ChessBoard.Board[::-1]
        self.BoardManager.PieceBoards.PieceBoards = np.ascontiguousarray(self.BoardManager.PieceBoards.PieceBoards[:,:,::-1])

        # Reverse the castling properties
        self.BoardManager.ChessBoard.Castling[:] = self.BoardManager.ChessBoard.Castling[::-1]
        self.BoardManager.ChessBoard.KingPositions[:] = self.BoardManager.ChessBoard.KingPositions[::-1]

        # Reverse player and opponent
        helperPlayer = self.BoardManager.CurrentOpponent
        self.BoardManager.CurrentOpponent = self.BoardManager.CurrentPlayer
        self.BoardManager.CurrentPlayer = helperPlayer

        # Increment ply number
        self.PlyNumber += 1
