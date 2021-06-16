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
    return delta

@njit(cache=True)
def checkMove(player,piece,piecePosition_i,delt,chessBoard,debug=False):
    playerSign,oppSign=player,functions.getOpponent(player)
    validMove,capturedPiece=False,0
    newPos=[piecePosition_i[0]+delt[0],piecePosition_i[1]+delt[1]]
    if(debug): print("DEBUG (moves (checkMove)):",functions.getPieceName(piece),":",piecePosition_i,"+",delt,"=",newPos)
    if(newPos[0]<0 or newPos[0]>7 or newPos[1]<0 or newPos[1]>7 or (playerSign*chessBoard[newPos[1]][newPos[0]]>0)):
        if(debug): print("DEBUG (moves (checkMove)): Newpos=",newPos,"for",functions.getPieceName(piece),"is invalid!")
        validMove=False 
        capturedPiece=0
    else:
        if piece<6 and piece>0:
            if(oppSign*chessBoard[newPos[1]][newPos[0]]>0):
                validMove=True
                capturedPiece=abs(chessBoard[newPos[1]][newPos[0]])
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
                inBetweenPos=[piecePosition_i[0],piecePosition_i[1]+1]
                if(chessBoard[inBetweenPos[1]][inBetweenPos[0]]==0 and chessBoard[newPos[1]][newPos[0]]==0):
                    validMove=True
                    capturedPiece=0
                else:
                    validMove=False
                    capturedPiece=0
            elif(delt==[1,1] or delt==[-1,1]): # capture move
                if(oppSign*chessBoard[newPos[1]][newPos[0]]>0):
                    validMove=True
                    capturedPiece=abs(chessBoard[newPos[1]][newPos[0]])
                else:
                    validMove=False
                    capturedPiece=0
            else:
                print("ERROR (moves (checkMove)): Player",player,"'s pawn move from piecePosition_i=",piecePosition_i,"with delt=",delt,"is neither normal nor capture move! => move not allowed!")
        else:
            validMove=False
            capturedPiece=0
    return newPos,validMove,capturedPiece

@njit(cache=True)
def appendToList(moveDict,piece,row,col,delta,moveID):
    piecePos_i=np.array([row,col], dtype=np.int64)
    movelist=[]
    for Delta in delta:
        for i in range(1,8):
            delt=typed.List([d*i for d in Delta])
            newPos=[piecePos_i[0]+delt[0],piecePos_i[1]+delt[1]]
            if(newPos[0]<0 or newPos[0]>7 or newPos[1]<0 or newPos[1]>7):
                break
            elif(piece==6):
                if(piecePos_i[1]==6 and newPos[1]==7): #Pawn promotion moves
                    movelist.append((2 ,piecePos_i[0], piecePos_i[1], newPos[0], newPos[1]))
                    moveDict[(2,        piecePos_i[0], piecePos_i[1], newPos[0], newPos[1])]=moveID
                    moveID+=1                                                                                
                    movelist.append((3, piecePos_i[0], piecePos_i[1], newPos[0], newPos[1]))
                    moveDict[(3,        piecePos_i[0], piecePos_i[1], newPos[0], newPos[1])]=moveID
                    moveID+=1                                                                                
                    movelist.append((4, piecePos_i[0], piecePos_i[1], newPos[0], newPos[1]))
                    moveDict[(4,        piecePos_i[0], piecePos_i[1], newPos[0], newPos[1])]=moveID
                    moveID+=1                                                                                
                    movelist.append((5, piecePos_i[0], piecePos_i[1], newPos[0], newPos[1]))
                    moveDict[(5,        piecePos_i[0], piecePos_i[1], newPos[0], newPos[1])]=moveID
                    moveID+=1
            else:
                movelist.append((0, piecePos_i[0], piecePos_i[1], newPos[0], newPos[1]))
                moveDict[(0,        piecePos_i[0], piecePos_i[1], newPos[0], newPos[1])]=moveID
                moveID+=1
            if(piece==6 or piece==5 or piece==1):
                break
            else:
                i+=1
    return moveID,movelist

@njit(cache=True)
def getAllMoves(moveDict):
    moveID=0
    moveDict[(1, 4, 0, 2, 0)]=moveID #castling short
    moveID+=1
    moveDict[(1, 4, 0, 6, 0)]=moveID #castling long
    moveID+=1
    moveList=[(1, 4, 0, 2, 0),(1, 4, 0, 6, 0)] #castling short, castling long

    for col in range(8):
        for row in range(8):
            delta1=np.array([[1,1],[-1,1],[1,-1],[-1,-1],[1,0],[-1,0],[0,1],[0,-1]],dtype=np.int64)  #Queen moves
            moveID,list1=appendToList(moveDict,0,row,col,delta1,moveID)
            moveList+=list1
            delta2=np.array([[1,2],[-1,2],[1,-2],[-1,-2],[2,1],[-2,1],[2,-1],[-2,-1]],dtype=np.int64)#Knight moves
            moveID,list2=appendToList(moveDict,5,row,col,delta2,moveID)
            moveList+=list2
            delta3=np.array([[1,1],[-1,1],[0,1]],dtype=np.int64)                                  #Pawn moves
            moveID,list3=appendToList(moveDict,6,row,col,delta3,moveID)
            moveList+=list3
    return moveList,moveDict
