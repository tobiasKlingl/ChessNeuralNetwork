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

        
    def findNextMove(self):
        self.BoardManager.ChessBoard.setIsPlayerInCheck()
        
        finished = False

        allMoves = self.BoardManager.getAllowedMovesList()
        numMoves = len(allMoves)
        print("lenAllMoves =", numMoves)
        
        if len(allMoves) > 0:
            if len(allMoves) <= 8:
                #Check if only king is left and opponent can still win!
                printInfo(self.NoOutputMode, "Less than 9 moves available! Check if game is decided.")

                playerSign = self.BoardManager.Players[self.BoardManager.CurrentPlayer]
                oppSign =    self.BoardManager.Players[self.BoardManager.CurrentOpponent]
                playerPieces =   []
                opponentPieces = []
                for row in self.BoardManager.ChessBoard.Board:
                    for col in row:
                        if playerSign * col > 0:
                            playerPieces.append(fn.getPieceName(playerSign * col))
                        elif oppSign * col > 0:
                            opponentPieces.append(fn.getPieceName(oppSign * col))

                if not self.NoOutputMode:
                    print("playerPieces =", playerPieces)

                if len(playerPieces) == 1:
                    if not self.NoOutputMode:
                        print("opponentPieces =", opponentPieces)

                    if len(opponentPieces) > 2 or len(opponentPieces) > 1 and ("rook" in opponentPieces or "queen" in opponentPieces):
                        for rand, move in enumerate(allMoves):
                            if move.CapturedPieceNum > 0:
                                moveInfoList = self.BoardManager.playMove(move, True)
                                if not self.NoOutputMode:
                                    evaluation = 1
                                    self.BoardManager.ChessBoard.printChessBoardWithInfo(moveID = move.getMoveID(), evaluation = evaluation, moveInfo = moveInfoList, colored = self.ColoredOutput)
                                return False, -1, 1
                        self.Winner = self.BoardManager.CurrentOpponent
                        return True, -1, 1
                    elif len(opponentPieces) == 2 and ("bishop" in opponentPieces or "knight" in opponentPieces):
                        self.Winner = "draw"
                        return True, -1, 1

                    
            #move =  max(allMoves, key = lambda move: move.MoveEvaluation)
            """
            maxProb = moveProbabilities[argMax]
            helper = np.empty_like(moveProbabilities)
            s = np.sum(moveProbabilities)
            probNormed = moveProbabilities/s
            np.round(moveProbabilities*1000,3,helper)
            prob_rounded = helper.astype(np.int32)
            """

            if self.GameMode == "SelfplayRandomVsRandom" or (self.GameMode == "SelfplayNetworkVsRandom" and self.BoardManager.CurrentPlayer == "black"):
                n_arg = random.randint(0, numMoves - 1)
                move = allMoves[n_arg]
                printInfo(self.NoOutputMode, "Random mover is moving now!")
            elif self.GameMode == "SelfplayNetworkVsNetwork" or (self.GameMode == "SelfplayNetworkVsRandom" and self.BoardManager.CurrentPlayer == "white"):
                n_arg = fn.rand_choice_nb(np.arange(numMoves), allMoves)
                move = allMoves[n_arg]
                printInfo(self.NoOutputMode, "Neural network is moving now!")

            if not self.NoOutputMode:
                print("numMoves =", numMoves, "; Move number n_arg =", n_arg, "chosen")
                print("moveID =", move.getMoveID())

            if __debug__:
                printDebug(" ".join(["rand =",rand]), fName = "nextMove")
                
            #if not self.NoOutputMode:
            #    fn.printMoves(self.CurentPlayer, allMoves, self.coloredOutput)

            moveInfoList = self.BoardManager.playMove(move, True)

            if not self.NoOutputMode:
                evaluation = 1
                self.BoardManager.ChessBoard.printChessBoardWithInfo(moveID = move.getMoveID(), evaluation = evaluation, moveInfo = moveInfoList, colored = self.ColoredOutput)
                print("self.BoardManager.ChessBoard.Castling =",  self.BoardManager.ChessBoard.Castling)
                print("self.BoardManager.ChessBoard.EnPassant =", self.BoardManager.ChessBoard.EnPassant)
        elif len(allMoves) == 0:
            if self.BoardManager.ChessBoard.IsPlayerInCheck:
                printInfo(self.NoOutputMode, "Player", self.BoardManager.CurrentPlayer, "is CHECKMATE!")
                self.Winner = self.BoardManager.CurrentOpponent
            elif not self.BoardManager.ChessBoard.IsPlayerInCheck:
                printInfo(self.NoOutputMode, "Player", self.BoardManager.CurrentPlayer, "has no more moves available => Remis.")
                self.Winner = "draw"

            finished = True
                
        posEval = 1 #evaluatePosition(boardpositions.ChessBoard)
        return finished, move.getMoveID(), posEval

