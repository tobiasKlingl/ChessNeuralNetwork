from Board import BoardManager
import numba as nb
from numba.experimental import jitclass




############################################################################################################################
#### jited BoardManager class ############################################################################################## 
############################################################################################################################

GameManagerSpecs = [
    ('ClassName',       nb.types.string),
    ('BoardManager',    BoardManager.class_type.instance_type),
    ('PlyNumber',       nb.int64),
    ('Finished',        nb.boolean),
    ('NoOutputMode',    nb.boolean),
    ('ColoredOutput',   nb.boolean),
]

@jitclass(GameManagerSpecs)
class GameManager(object):

    def __init__(self, boardManager, noOutputMode, coloredOutput) -> None:
        self.ClassName =       "GameManager"
        self.BoardManager =    boardManager
        self.PlyNumber =       1
        self.Finished =        False
        self.NoOutputMode =    noOutputMode
        self.ColoredOutput =   coloredOutput

        
    def findNextMove(self):
        self.BoardManager.ChessBoard.setIsPlayerInCheck()

        allMoves, moveProbabilities = self.BoardManager.getAllowedMovesList()
        finished = False
        moveID = -1

        print("lenAllMoves =",len(allMoves))
        
        """
        if len(outOfCheckMoves) > 0:
            if len(outOfCheckMoves) <= 8:
                #Check if only king is left and opponent can still win!
                printInfo(self.NoOutputMode, "Less than 9 moves available! Check if game is decided.")

                playerSign = self.BoardManager.Players[self.BoardManager.CurrentPlayer]
                oppSign =    self.BoardManager.Players[self.BoardManager.CurrentOpponent]
                playerPieces =   []
                opponentPieces = []
                for row in self.BoardManager.ChessBoard:
                    for col in row:
                        if playerSign * col > 0:
                            playerPieces.append(functions.getPieceName(playerSign * col))
                        elif oppSign * col > 0:
                            opponentPieces.append(functions.getPieceName(oppSign * col))
                            
                printInfo(self.NoOutputMode, "playerPieces=",playerPieces)

                if len(playerPieces) == 1:
                    printInfo(self.NoOutputMode, "opponentPieces=",opponentPieces)

                    if len(opponentPieces) > 2 or len(opponentPieces) > 1 and ("rook" in opponentPieces or "queen" in opponentPieces):
                        for rand, move in enumerate(outOfCheckMoves):
                            if move[6] != 0:
                                moveInfoList, captPiece = self.BoardManager.playMove(move, True)
                                if not self.NoOutputMode:
                                    printChessBoard(self.BoardManager.ChessBoard, self.BoardManager.CurrentPlayer, move[10], -1, moveInfoList, self.ColoredOutput)
                                return False, -1
                        self.setWinner(self.BoardManager.CurrentOpponent)
                        return True, -1
                    elif len(opponentPieces) == 2 and ("bishop" in opponentPieces or "knight" in opponentPieces):
                        self.setWinner("")
                        return True, -1

            maxNum = len(outOfCheckMoves)
            argMax = np.argmax(moveProbabilities)
            maxProb = moveProbabilities[argMax]
            helper = np.empty_like(moveProbabilities)
            s = np.sum(moveProbabilities)
            probNormed = moveProbabilities/s
            np.round(moveProbabilities*1000,3,helper)
            prob_rounded = helper.astype(np.int32)

            if self.GameMode == "SelfplayRandomVsRandom" or (self.GameMode == "SelfplayNetworkVsRandom" and self.CurentPlayer == "black"):
                rand = random.randint(0, maxNum-1)
                move = outOfCheckMoves[rand]
                if(noOutputMode == False):
                    print("Random mover is moving now!")
                    print("maxNum,rand=", maxNum, rand)
                    print("move[10]=", move[10])
                moveID = move[10]
            elif self.GameMode == "SelfplayNetworkVsNetwork" or (self.GameMode == "SelfplayNetworkVsRandom" and self.CurentPlayer == "white"):
                #rand=np.argmax(moveProbabilities)
                rand=rand_choice_nb(np.arange(maxNum),probNormed)
                move=outOfCheckMoves[rand]
                if(noOutputMode==False):
                    print("Neural network is moving now!")
                    print("maxNum,rand=",maxNum,rand)
                    print("move[10]=",move[10])
                moveID=move[10]
                #np.random.choice(np.arange(maxNum), p=probNormed)
                #rand=random.randint(0,maxNum-1)
            #else:
            #    rand=random.randint(0,maxNum)
            #    move=outOfCheckMoves[rand]
            #    moveID=move[10]
            if(debug): print("(functions (nextMove)): rand=",rand)
            if not self.NoOutputMode:
                printMoves(self.CurentPlayer, outOfCheckMoves, colored, prob_rounded)
            moveInfoList, captPiece = self.BoardManager.playMove(move, True)
            if not self.NoOutputMode:
                printChessBoard(boardpositions.ChessBoard, self.CurentPlayer, move[10], prob_rounded[rand], moveInfoList, colored)
                #print("boardpositions.Castling=",boardpositions.Castling)
                #print("boardpositions.EnPassant=",boardpositions.EnPassant)
        elif len(outOfCheckMoves) == 0 and self.IsPlayerInCheck == True:
            if not self.NoOutputMode:
                print("Player", self.BoardManager.CurrentPlayer, "is CHECKMATE!")
            self.setWinner(self.BoardManager.CurrentOpponent)
            finished = True
        elif(len(outOfCheckMoves)==0 and self.IsPlayerInCheck == False):
            if not self.NoOutputMode:
                print("Player", self.BoardManager.CurrentPlayer, "has no more moves available => Remis.")
            boardpositions.setWinner(0.5)
            finished=True
        posEval=evaluatePosition(boardpositions.ChessBoard)
        return finished, moveID, posEval
        """
