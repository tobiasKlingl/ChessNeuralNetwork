import numpy as np
import functions

def getAllowedMovesForPiece(player,piece,piecePosition_i,chessBoard,onlyCaptureMoves,debug=False):
    allowedMoves=[]
    delta=getBasicMoves(piece,piecePosition_i,onlyCaptureMoves,debug)
    if(debug): print("DEBUG (moves (getAllowedMovesForPiece)): piece=",piece)
    for Delta in delta:
        validMove=True
        captureMove=False
        for i in range(1,8):
            if validMove==True and captureMove==False:
                Del=[d*i for d in Delta]
                if(debug): print("DEBUG (moves (getAllowedMovesForPiece)): Del=",Del)
                newPos,validMove,captureMove=checkMove(player,piece,piecePosition_i,Del,chessBoard,debug)
                if(validMove==True):
                    if(piece=="pawn" and piecePosition_i[1]==6 and newPos[1]==7):
                        allowedMoves.append([piece,"knight",piecePosition_i,newPos,captureMove]) # [piece before, piece after, pos before, pos after, captured?]
                        allowedMoves.append([piece,"bishop",piecePosition_i,newPos,captureMove])
                        allowedMoves.append([piece,"rook"  ,piecePosition_i,newPos,captureMove])
                        allowedMoves.append([piece,"queen" ,piecePosition_i,newPos,captureMove])
                    else:
                        allowedMoves.append([piece,piece,piecePosition_i,newPos,captureMove])
                    if(piece=="pawn" or piece=="knight" or piece=="king"):
                        break
                else:
                    break
            else:
                break
    if(debug): print("DEBUG (moves (getAllowedMovesForPiece)): AllowedMoves=",allowedMoves)
    return allowedMoves

def getBasicMoves(piece,piecePosition_i,onlyCaptureMoves,debug=False):
    delta=[]
    if piece=="pawn":
        delta=[[1,1],[-1,1]]
        if(piecePosition_i[1]==1 and onlyCaptureMoves==False):
            delta.append([0,1])
            delta.append([0,2])
        elif(onlyCaptureMoves==False):
            delta.append([0,1])
    elif(piece=="knight"):delta=[[1,2],[-1,2],[1,-2],[-1,-2],[2,1],[-2,1],[2,-1],[-2,-1]]
    elif(piece=="bishop"):delta=[[1,1],[-1,1],[1,-1],[-1,-1]]
    elif(piece=="rook")  :delta=[[1,0],[-1,0],[0,1],[0,-1]]
    elif(piece=="queen" or piece=="king"): delta=[[1,1],[-1,1],[1,-1],[-1,-1],[1,0],[-1,0],[0,1],[0,-1]]
    else:print("(ERROR (moves (getBasicMoves)): Piece unknown!")
    return delta

def checkMove(player,piece,piecePosition_i,delt,chessBoard,debug=False):
    playerColor=  functions.getPlayerColor(player)
    opponentColor=functions.getPlayerColor(functions.getOpponent(player))
    validMove=False
    captureMove=False
    returnArray=[]
    newPos=np.add(piecePosition_i,delt).tolist()
    if(debug): print("DEBUG (moves (checkMove)): piece,piecePosition_i,delt,newPos=",piece,piecePosition_i,delt,newPos)
    if(newPos[0]<0 or newPos[0]>7 or newPos[1]<0 or newPos[1]>7 or (playerColor in chessBoard[newPos[1]][newPos[0]])):
        if(debug): print("DEBUG (moves (checkMove)): Newpos=",newPos,"for",piece,"is invalid!")
        returnArray=[newPos,False,False] #newPos,validMove,captureMove
    else:
        if piece!="pawn":
            if(opponentColor in chessBoard[newPos[1]][newPos[0]]):
                returnArray=[newPos,True,True]
            else:
                returnArray=[newPos,True,False]
        elif piece=="pawn":
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
