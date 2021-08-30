import Board
from moves import mL,mD
from functions import printInfo, printDebug, printError
import numba as nb
from numba.experimental import jitclass

noOutputMode = False

print("mL =",mL)
print("mL =",len(mL))
print("")
print("mD =",mD)





def play():
    # initialize BoardManager
    chessBoard = Board.ChessBoard(noOutputMode)
    boardManager=Board.BoardManager(chessBoard, noOutputMode)
    boardManager.initializePieceBoards()
    boardManager.setPieces()
    boardManager.ChessBoard.printChessBoard()

    initialPlyNumber = boardManager.PlyNumber
    initialPlayer =    boardManager.CurrentPlayer
    initialOpponent =  boardManager.CurrentOpponent

    printInfo(noOutputMode, "initialPlyNumber, initialPlayer, initialOpponent =", str(initialPlyNumber), initialPlayer, initialOpponent)




    
play()

