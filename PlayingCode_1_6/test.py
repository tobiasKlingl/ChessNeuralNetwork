import os
import numba as nb
from numba.experimental import jitclass

import Board
import Game
from moves import getMoveDict
from functions import printInfo, printDebug, printError
    

@nb.njit(cache = True)
def play(gameNumber):
    noOutputMode = False
    coloredOutput = False
    searchDepth = 20
    
    mD = getMoveDict()

    boardInformation = Board.BoardInformation(noOutputMode, coloredOutput)
    chessBoard = Board.ChessBoard(boardInformation)
    pieceBoards = Board.PieceBoards(boardInformation)
    boardManager = Board.BoardManager(boardInformation, chessBoard, pieceBoards, mD)
    boardManager.initialize()
    boardManager.setPieces()

    gameManager = Game.GameManager(boardManager, noOutputMode, coloredOutput)
    finishedInOne = 0

    initialPlyNumber = gameManager.PlyNumber
    initialPlayer =    boardInformation.CurrentPlayer
    initialOpponent =  boardInformation.CurrentOpponent

    printInfo(noOutputMode, "initialPlyNumber, initialPlayer, initialOpponent =", str(initialPlyNumber), initialPlayer, initialOpponent)

    while not gameManager.Finished:
        currentPlayer = boardInformation.CurrentPlayer
        
        if coloredOutput:
            printInfo(noOutputMode, "\nGame number " + str(gameNumber) + ": \033[1;31;48m###### Ply ", str(gameManager.PlyNumber), " (player:", currentPlayer + ") ######\033[1;37;48m")    
        else:
            printInfo(noOutputMode, "\nGame number " + str(gameNumber) + ": ###### Ply ",              str(gameManager.PlyNumber), " (player:", currentPlayer + ") ######")

        moveID, moveEval = gameManager.findNextMove()

        if gameManager.PlyNumber == initialPlyNumber:
            print("moveID =", moveID)
            finishedInOne = gameManager.Finished
            firstMoveID = moveID

        if gameManager.PlyNumber >= searchDepth:
            gameManager.Winner = "draw"
            printInfo(noOutputMode, "Game number " + str(gameNumber), ": INFO: Anticipated search depth= ", str(searchDepth)," reached. Get Evaluation ...")
            break
        else:
            gameManager.nextPly()

    winner = gameManager.Winner
    textColor, resetColor = "", ""

    if coloredOutput == True:
        textColor, resetColor = "\033[1;31;49m", "\033[1;37;49m"

    if gameManager.Winner == initialPlayer:
        won = 1.
        #if noOutput==False:
        print(textColor + str(gameNumber) + ": Player", winner, "won in", gameManager.PlyNumber, "plys!", resetColor)
    elif gameManager.Winner == initialOpponent:
        won = 0.
        #if noOutput==False:
        print(textColor + str(gameNumber) + ": Player", winner, "won in", gameManager.PlyNumber, "plys!", resetColor)
    elif gameManager.Winner == "draw":
        won = 0.5
        #if noOutput==False:
        print(textColor + str(gameNumber) + ": Game ended remis!", resetColor)
    else:
        won = -99.
        print(str(gameNumber) + ": ERROR: Unknown value for winner! winner =", winner)

    #return firstMoveID, won, finishedInOne


@nb.njit(cache = True)
def playAllGames():
    nGames = 1
    for gameNum in range(nGames):
        play(gameNum)

        
if __name__ == "__main__":
    playAllGames()
