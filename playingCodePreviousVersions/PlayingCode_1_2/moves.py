import numpy as np
import functions

def getBasicMoves(piece,piecePosition_i,onlyCaptureMoves,debug=False):
    if piece=="p":
        delta=[(1,1),(-1,1)]
        if onlyCaptureMoves==False:
            delta.append((0,1))
            if piecePosition_i[1]==1:
               delta.append((0,2))
    elif(piece=="n"):delta=((1,2),(-1,2),(1,-2),(-1,-2),(2,1),(-2,1),(2,-1),(-2,-1))
    elif(piece=="b"):delta=((1,1),(-1,1),(1,-1),(-1,-1))
    elif(piece=="r"):delta=((1,0),(-1,0),( 0,1),( 0,-1))
    elif(piece=="q" or piece=="k"): delta=((1,1),(-1,1),(1,-1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1))
    else: print("(ERROR (moves (getBasicMoves)): Piece unknown!")
    return delta

def checkMove(player,piece,piecePosition_i,delt,chessBoard,debug=False):
    playerColor,opponentColor=functions.getPlayerColor(player),functions.getPlayerColor(functions.getOpponent(player))
    validMove,captureMove=False,False
    returnArray=[]
    newPos=np.add(piecePosition_i,delt).tolist()
    if(debug): print("DEBUG (moves (checkMove)): piece,piecePosition_i,delt,newPos=",piece,piecePosition_i,delt,newPos)
    if(newPos[0]<0 or newPos[0]>7 or newPos[1]<0 or newPos[1]>7 or (playerColor in chessBoard[newPos[1]][newPos[0]])):
        if(debug): print("DEBUG (moves (checkMove)): Newpos=",newPos,"for",piece,"is invalid!")
        returnArray=[newPos,False,False] #newPos,validMove,captureMove
    else:
        if piece!="p":
            if(opponentColor in chessBoard[newPos[1]][newPos[0]]):
                returnArray=[newPos,True,True]
            else:
                returnArray=[newPos,True,False]
        elif piece=="p":
            if(delt==[0,1]): # normal move
                if(chessBoard[newPos[1]][newPos[0]]=="  "):
                    returnArray=[newPos,True,False]
                else:
                    returnArray=[newPos,False,False]
            elif(delt==[0,2]): # normal move (Check also in-between position)
                inBetweenPos=np.add(piecePosition_i,[0,1]).tolist()
                if(chessBoard[inBetweenPos[1]][inBetweenPos[0]]=="  " and chessBoard[newPos[1]][newPos[0]]=="  "):
                    returnArray=[newPos,True,False]
                else:
                    returnArray=[newPos,False,False]
            elif(delt==[1,1] or delt==[-1,1]): # capture move
                if(opponentColor in chessBoard[newPos[1]][newPos[0]]):
                    returnArray=[newPos,True,True]
                else:
                    returnArray=[newPos,False,False]
            else:
                print("ERROR (moves (checkMove)): Player",player,"'s pawn move from piecePosition_i=",piecePosition_i,"with delt=",delt,"is neither normal nor capture move! => move not allowed!")
        else:
            returnArray=[newPos,False,False]
    return returnArray[0],returnArray[1],returnArray[2]
