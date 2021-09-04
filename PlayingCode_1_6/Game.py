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


    def getMoveNumToPlay(self, allMoves) -> nb.int64:
        numMoves = len(allMoves)

        if self.GameMode == "SelfplayRandomVsRandom" or (self.GameMode == "SelfplayNetworkVsRandom" and self.BoardManager.BoardInfo.CurrentPlayer == "black"):
            printInfo(self.NoOutputMode, "Random mover is moving now!")
            n_move = fn.rand_choice_nb(np.arange(numMoves), allMoves)
            #n_move = random.randint(0, numMoves - 1)
        elif self.GameMode == "SelfplayNetworkVsNetwork" or (self.GameMode == "SelfplayNetworkVsRandom" and self.BoardManager.BoardInfo.CurrentPlayer == "white"):
            printInfo(self.NoOutputMode, "Neural network is moving now!")
            n_move = fn.rand_choice_nb(np.arange(numMoves), allMoves)

        printInfo(self.NoOutputMode, " ".join(["numMoves =", str(numMoves), "; Move number n_move =", str(n_move), "chosen"]))
            
        return n_move

    
    def gameEnded(self):
        if self.BoardManager.BoardInfo.IsPlayerInCheck:
            printInfo(self.NoOutputMode, "Player", self.BoardManager.BoardInfo.CurrentPlayer, "is CHECKMATE!")
            self.Winner = self.BoardManager.BoardInfo.CurrentOpponent
        elif not self.BoardManager.BoardInfo.IsPlayerInCheck:
            printInfo(self.NoOutputMode, "Player", self.BoardManager.BoardInfo.CurrentPlayer, "has no more moves available => Remis.")
            self.Winner = "draw"

        self.Finished = True

        
    def nextMove(self) -> (nb.int64, nb.float64):
        self.BoardManager.ChessBoard.setIsPlayerInCheck()
        if self.BoardManager.BoardInfo.IsPlayerInCheck:
            printInfo(self.NoOutputMode, self.BoardManager.BoardInfo.CurrentPlayer, "is in CHECK!")
        
        self.Finished = False
        moveID = -1
        moveEval = -1

        allMoves = self.BoardManager.getAllowedMovesList()
        numMoves = len(allMoves)
        printInfo(self.NoOutputMode, "Number of available moves =", str(numMoves))

        if numMoves == 0: #No more moves available
            self.gameEnded()
        elif numMoves > 0:
            if numMoves <= 8:
                #Check if only king is left and opponent can still win!
                if not self.BoardManager.BoardInfo.NoOutputMode:
                    print("Less than 9 moves available! Check if game is decided.")
                playerPieces, opponentPieces = self.BoardManager.ChessBoard.getPiecesOnBoard()
                
                if len(playerPieces) == 1:
                    if len(opponentPieces) > 2 or len(opponentPieces) > 1 and ("rook" in opponentPieces or "queen" in opponentPieces):
                        for rand, move in enumerate(allMoves):
                            if move.CapturedPieceNum > 0:
                                moveInfoList = self.BoardManager.playMove(move, writeMoveInfoList = True)
                                if not self.NoOutputMode:
                                    self.BoardManager.ChessBoard.printChessBoardWithInfo(moveID = move.getMoveID(), evaluation = move.Evaluation, moveInfo = moveInfoList, colored = self.ColoredOutput)
                                self.Finished = False
                                return -1, 1

                        self.Winner = self.BoardManager.BoardInfo.CurrentOpponent
                        self.Finished = True
                        return -1, 1
                    elif len(opponentPieces) == 2 and ("bishop" in opponentPieces or "knight" in opponentPieces):
                        self.Winner = "draw"
                        self.Finished = True
                        return -1, 1

            if not self.NoOutputMode:
                fn.printMoves(self.BoardManager.BoardInfo.CurrentPlayer, allMoves, self.ColoredOutput, self.NoOutputMode)
            
            n_move = self.getMoveNumToPlay(allMoves)
            move = allMoves[n_move]
            moveID = move.getMoveID()
            
            moveInfoList = self.BoardManager.playMove(move, True)

            if not self.NoOutputMode:
                self.BoardManager.ChessBoard.printChessBoardWithInfo(moveID = moveID, evaluation = move.Evaluation, moveInfo = moveInfoList, colored = self.ColoredOutput)
                print("BoardManager.BoardInfo.Castling =",  self.BoardManager.BoardInfo.Castling)
                print("BoardManager.BoardInfo.EnPassant =", self.BoardManager.BoardInfo.EnPassant)
                
        return moveID, move.Evaluation


    def increasePlyNumber(self):
        # Reverse the board
        if self.BoardManager.BoardInfo.Reversed:
            self.BoardManager.BoardInfo.Reversed = False
        elif not self.BoardManager.BoardInfo.Reversed:
            self.BoardManager.BoardInfo.Reversed = True

        self.BoardManager.ChessBoard.Board[:] = self.BoardManager.ChessBoard.Board[::-1]
        self.BoardManager.PieceBoards.BitBoards = np.ascontiguousarray(self.BoardManager.PieceBoards.BitBoards[:,:,::-1])

        # Reverse player and opponent
        helperPlayer = self.BoardManager.BoardInfo.CurrentOpponent
        self.BoardManager.BoardInfo.CurrentOpponent = self.BoardManager.BoardInfo.CurrentPlayer
        self.BoardManager.BoardInfo.CurrentPlayer = helperPlayer

        # Increment ply number
        self.PlyNumber += 1
