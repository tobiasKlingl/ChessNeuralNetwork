import Board
import game
from moves import MoveDictionary
from functions import printInfo, printDebug, printError
import numba as nb
from numba.experimental import jitclass

noOutputMode = False
coloredOutput = False


moveDict = MoveDictionary()
mL = moveDict.MoveList
mD = moveDict.MoveDict

"""
print("mL =",mL)
print("mL =",len(mL))
print("")
print("mD =",mD)
"""

def play(gameNumber):
    # initialize BoardManager
    chessBoard = Board.ChessBoard(noOutputMode, coloredOutput)
    boardManager=Board.BoardManager(chessBoard, mD, noOutputMode, coloredOutput)
    boardManager.initializePieceBoards()
    boardManager.setPieces()

    initialPlyNumber = boardManager.PlyNumber
    initialPlayer =    boardManager.CurrentPlayer
    initialOpponent =  boardManager.CurrentOpponent

    printInfo(noOutputMode, "initialPlyNumber, initialPlayer, initialOpponent =", str(initialPlyNumber), initialPlayer, initialOpponent)

    gameManager = game.GameManager(boardManager, noOutputMode, coloredOutput)
    finishedInOne = 0

    while(gameManager.Finished == False):
        currentPlayer = gameManager.BoardManager.CurrentPlayer
        
        if coloredOutput:
            printInfo(noOutputMode, "\nGame number " + str(gameNumber) + ": \033[1;31;48m###### Ply ", str(gameManager.PlyNumber), " (player:", currentPlayer + ") ######\033[1;37;48m")    
        else:
            printInfo(noOutputMode, "\nGame number " + str(gameNumber) + ": ###### Ply ",              str(gameManager.PlyNumber), " (player:", currentPlayer + ") ######")

        gameManager.Finished, moveID, posEval = gameManager.findNextMove()
        gameManager.Finished = True
        
play(1)

