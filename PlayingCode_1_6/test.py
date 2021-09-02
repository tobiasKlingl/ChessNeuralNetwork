import os
import numba as nb
from numba.experimental import jitclass

import Board
import game
from moves import getMoveDict
from functions import printInfo, printDebug, printError

#os.environ['NUMBA_DISABLE_JIT'] = '1'

noOutputMode = False
coloredOutput = False


mD = getMoveDict()
#print("mD =", mD)

def play(gameNumber):
    # initialize BoardManager
    chessBoard =  Board.ChessBoard(noOutputMode,  coloredOutput)
    pieceBoards = Board.PieceBoards(noOutputMode)
    boardManager = Board.BoardManager(chessBoard, pieceBoards, mD, noOutputMode, coloredOutput)
    boardManager.initializePieceBoards()
    boardManager.setPieces()

    gameManager = game.GameManager(boardManager, noOutputMode, coloredOutput)
    finishedInOne = 0

    initialPlyNumber = gameManager.PlyNumber
    initialPlayer =    boardManager.CurrentPlayer
    initialOpponent =  boardManager.CurrentOpponent

    printInfo(noOutputMode, "initialPlyNumber, initialPlayer, initialOpponent =", str(initialPlyNumber), initialPlayer, initialOpponent)

    while(gameManager.Finished == False):
        currentPlayer = gameManager.BoardManager.CurrentPlayer
        
        if coloredOutput:
            printInfo(noOutputMode, "\nGame number " + str(gameNumber) + ": \033[1;31;48m###### Ply ", str(gameManager.PlyNumber), " (player:", currentPlayer + ") ######\033[1;37;48m")    
        else:
            printInfo(noOutputMode, "\nGame number " + str(gameNumber) + ": ###### Ply ",              str(gameManager.PlyNumber), " (player:", currentPlayer + ") ######")

        gameManager.Finished, moveID, posEval = gameManager.findNextMove()
        gameManager.Finished = True
        
play(1)

