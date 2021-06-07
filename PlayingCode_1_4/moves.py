import numpy as np
import functions
from numba import jit,njit,types,int64,typed

@njit(cache=True)
def getBasicMoves(piece,piecePosition_i,onlyCaptureMoves,debug=False):
    if piece==6:
        delta=[[1,1],[-1,1]]
        if onlyCaptureMoves==False:
            delta=[[1,1],[-1,1],[0,1]]
            if piecePosition_i[1]==1:
                delta=[[1,1],[-1,1],[0,1],[0,2]]
    elif(piece==5):delta=[[1,2],[-1,2],[1,-2],[-1,-2],[2,1],[-2,1],[2,-1],[-2,-1]]
    elif(piece==4):delta=[[1,1],[-1,1],[1,-1],[-1,-1]]
    elif(piece==3):delta=[[1,0],[-1,0],[ 0,1],[ 0,-1]]
    elif(piece==2 or piece==1): delta=[[1,1],[-1,1],[1,-1],[-1,-1],[1,0],[-1,0],[0,1],[0,-1]]
    else:
        delta=[[np.int64(x),np.int64(x)] for x in range(0)]
        print("[ERROR [moves [getBasicMoves]]: Piece unknown!")
    """
    if piece=="p":
        delta=typed.List(( typed.List((1,1)), typed.List((-1,1)) ))
        if onlyCaptureMoves==False:
            delta=typed.List(( typed.List((1,1)), typed.List((-1,1)), typed.List((0,1)) ))
            if piecePosition_i[1]==1:
                delta=typed.List(( typed.List((1,1)),typed.List((-1,1)),typed.List((0,1)),typed.List((0,2)) ))
    elif(piece=="n"):delta=typed.List(( typed.List((1,2)),typed.List((-1,2)),typed.List((1,-2)),typed.List((-1,-2)),typed.List((2,1)),typed.List((-2,1)),typed.List((2,-1)),typed.List((-2,-1)) ))
    elif(piece=="b"):delta=typed.List(( typed.List((1,1)),typed.List((-1,1)),typed.List((1,-1)),typed.List((-1,-1)) ))
    elif(piece=="r"):delta=typed.List(( typed.List((1,0)),typed.List((-1,0)),typed.List(( 0,1)),typed.List(( 0,-1)) ))
    elif(piece=="q" or piece=="k"): delta=typed.List(( typed.List((1,1)),typed.List((-1,1)),typed.List((1,-1)),typed.List((-1,-1)),typed.List((1,0)),typed.List((-1,0)),typed.List((0,1)),typed.List((0,-1)) ))
    """
    return delta

@njit(cache=True)
def checkMove(player,piece,piecePosition_i,delt,chessBoard,debug=False):
    playerSign,oppSign=player,functions.getOpponent(player)
    validMove,capturedPiece=False,0
    newPos=[piecePosition_i[0]+delt[0],piecePosition_i[1]+delt[1]]
    if(debug): print("DEBUG (moves (checkMove)): piece,piecePosition_i,delt,newPos=",functions.getPieceName(piece),piecePosition_i,delt,newPos)
    if(newPos[0]<0 or newPos[0]>7 or newPos[1]<0 or newPos[1]>7 or (playerSign*chessBoard[newPos[1]][newPos[0]]>0)):
        if(debug): print("DEBUG (moves (checkMove)): Newpos=",newPos,"for",functions.getPieceName(piece),"is invalid!")
        validMove=False 
        capturedPiece=0
    else:
        if piece<6 and piece>0:
            if(oppSign*chessBoard[newPos[1]][newPos[0]]>0):
                validMove=True
                capturedPiece=chessBoard[newPos[1]][newPos[0]]
            else:
                validMove=True
                capturedPiece=0
        elif piece==6:
            if(delt==[0,1]): # normal move
                if(chessBoard[newPos[1]][newPos[0]]==0):
                    validMove=True
                    capturedPiece=0
                else:
                    validMove=False
                    capturedPiece=0
            elif(delt==[0,2]): # normal move (Check also in-between position)
                inBetweenPos=[piecePosition_i[0],piecePosition_i[1]+1] #np.add(piecePosition_i,[0,1]).tolist()
                if(chessBoard[inBetweenPos[1]][inBetweenPos[0]]==0 and chessBoard[newPos[1]][newPos[0]]==0):
                    validMove=True
                    capturedPiece=0
                else:
                    validMove=False
                    capturedPiece=0
            elif(delt==[1,1] or delt==[-1,1]): # capture move
                if(oppSign*chessBoard[newPos[1]][newPos[0]]>0):
                    validMove=True
                    capturedPiece=chessBoard[newPos[1]][newPos[0]]
                else:
                    validMove=False
                    capturedPiece=0
            else:
                print("ERROR (moves (checkMove)): Player",player,"'s pawn move from piecePosition_i=",piecePosition_i,"with delt=",delt,"is neither normal nor capture move! => move not allowed!")
        else:
            validMove=False
            capturedPiece=0
    return newPos,validMove,capturedPiece
