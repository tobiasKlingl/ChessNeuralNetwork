import os
from timeit import default_timer as timer
import numba as nb
from numba.experimental import jitclass

import Board
import Game
from moves import getMoveDict
import functions as fn
from functions import printInfo, printDebug, printError

#os.environ["NUMBA_DISABLE_JIT"] = "1"

@nb.njit(cache = True)
def settings() -> (nb.int64, nb.boolean, nb.boolean, nb.int64):
    nGames = 1
    noOutputMode = False
    coloredOutput = True #False
    searchDepth = 500

    return nGames, noOutputMode, coloredOutput, searchDepth


@nb.njit(cache = True)
def playAllGames():
    nGames, noOutputMode, coloredOutput, searchDepth = settings()

    for gameNum in range(nGames):
        play(gameNum, noOutputMode, coloredOutput, searchDepth)


@nb.njit(cache = True)
def play(gameNumber, noOutputMode, coloredOutput, searchDepth):
    mD = getMoveDict()

    boardInfo =    Board.BoardInfo(noOutputMode, coloredOutput)
    chessBoard =   Board.ChessBoard(boardInfo)
    pieceBoards =  Board.PieceBoards(boardInfo)
    boardManager = Board.BoardManager(boardInfo, chessBoard, pieceBoards, mD)
    gameManager =  Game.GameManager(boardManager, noOutputMode, coloredOutput)

    finishedInOne = 0
    firstMoveID = -1

    initialPlyNumber = gameManager.PlyNumber
    initialPlayer =    boardInfo.CurrentPlayer
    initialOpponent =  boardInfo.CurrentOpponent
    printInfo(noOutputMode, "initialPlyNumber, initialPlayer, initialOpponent =", str(initialPlyNumber), initialPlayer, initialOpponent)

    ### Playing the game
    while not gameManager.Finished:
        if coloredOutput:
            printInfo(noOutputMode, "\nGame number " + str(gameNumber) + ": \033[1;31;48m###### Ply ", str(gameManager.PlyNumber), " (player:", boardInfo.CurrentPlayer + ") ######\033[1;37;48m")    
        else:
            printInfo(noOutputMode, "\nGame number " + str(gameNumber) + ": ###### Ply ",              str(gameManager.PlyNumber), " (player:", boardInfo.CurrentPlayer + ") ######")

        moveID, moveEval = gameManager.nextMove()

        if gameManager.PlyNumber == initialPlyNumber:
            finishedInOne = gameManager.Finished
            firstMoveID = moveID

        if gameManager.PlyNumber >= searchDepth:
            gameManager.Winner = "draw"
            printInfo(noOutputMode, "Game number " + str(gameNumber), ": INFO: Anticipated search depth= ", str(searchDepth)," reached. Get Evaluation ...")
            break
        else:
            gameManager.increasePlyNumber()

    outputVal = fn.getOutputVal(gameNumber, gameManager.PlyNumber, initialPlayer, initialOpponent, gameManager.Winner, coloredOutput, noOutputMode)

    print("gameManager.Winner, gameManager.PlyNumber, firstMoveID, outputVal, finishedInOne =", gameManager.Winner, gameManager.PlyNumber, firstMoveID, outputVal, finishedInOne)


        
if __name__ == "__main__":
    start_time = timer()
    playAllGames()
    print(timer() - start_time, "seconds")


