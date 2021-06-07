import numpy as np
import random
from timeit import default_timer as timer
from numba import jit,njit,types,int64,typed

@njit(cache=True)
def getPlayerNumerator(player):
    if  (player==+1): numerator=0
    elif(player==-1): numerator=1
    else:print("(ERROR (functions (getPlayerColor)): invalid argument for player=",player)
    return numerator

@njit(cache=True)
def getOpponent(player):
    if  (player==+1):opponent=-1
    elif(player==-1):opponent=+1
    else:print("(ERROR (function (getOpponent): invalid argument for player=",player)
    return opponent

@njit(cache=True)
def getReadablePosition(player,pos,debug=False):
    if  (pos[0]==0): xpos="A"
    elif(pos[0]==1): xpos="B"
    elif(pos[0]==2): xpos="C"
    elif(pos[0]==3): xpos="D"
    elif(pos[0]==4): xpos="E"
    elif(pos[0]==5): xpos="F"
    elif(pos[0]==6): xpos="G"
    elif(pos[0]==7): xpos="H"
    else:
        xpos=""
        print("ERROR (functions (getReadablePosition)): pos[0]=",pos[0]," is unknown!")
    if  (player==+1): ypos=str(pos[1]+1)
    elif(player==-1): ypos=str(7-pos[1]+1)
    else:
        ypos=""
        print("ERROR (functions (getReadablePosition)): player=",player," is unknown!")
    readablePosition=xpos+ypos
    if(debug): print("DEBUG (functions (getReadablePosition)): Converted position:",pos,"into:",readablePosition)
    return readablePosition

@njit(cache=True)
def getPieceName(pieceAbb):
    name=""
    pieceNumber=abs(pieceAbb)
    if   pieceNumber==1: name=="k" 
    elif pieceNumber==2: name=="q" 
    elif pieceNumber==3: name=="r" 
    elif pieceNumber==4: name=="b" 
    elif pieceNumber==5: name=="n" 
    elif pieceNumber==6: name=="p" 
    else:print("ERROR (BoardPositions (getPieceNumber)): PieceNumber",pieceNumber,"unknown!")
    return name

@njit(cache=True)
def getPieceUnicode(piece,colored):
    pieceUnicode=" "
    pieceAbb=abs(piece)
    if(colored==True):
        if  (pieceAbb==1): pieceUnicode='\u265A'
        elif(pieceAbb==2): pieceUnicode='\u265B'
        elif(pieceAbb==3): pieceUnicode='\u265C'
        elif(pieceAbb==4): pieceUnicode='\u265D'
        elif(pieceAbb==5): pieceUnicode='\u265E'
        elif(pieceAbb==6): pieceUnicode='\u265F'
    else:
        if(piece>0): x=6
        elif(piece<0): x=0
        if  (pieceAbb==1): pieceUnicode="".join(chr(9812+pieceAbb-1+x))
        elif(pieceAbb==2): pieceUnicode="".join(chr(9812+pieceAbb-1+x))
        elif(pieceAbb==3): pieceUnicode="".join(chr(9812+pieceAbb-1+x))
        elif(pieceAbb==4): pieceUnicode="".join(chr(9812+pieceAbb-1+x))
        elif(pieceAbb==5): pieceUnicode="".join(chr(9812+pieceAbb-1+x))
        elif(pieceAbb==6): pieceUnicode="".join(chr(9812+pieceAbb-1+x))
    return pieceUnicode

@njit(cache=True)
def createPieceListforPrintOut(readableMoveList,prob,colored=False,debug=False):
    sortedList=[]
    pawnMoves  ="  PAWN: "
    knightMoves="  KNIGHT: "
    bishopMoves="  BISHOP: "
    rookMoves  ="  ROOK: "
    queenMoves ="  QUEEN: "
    kingMoves  ="  KING: "
    for ID,move in enumerate(readableMoveList):
        if colored:
            moveWithID="\033[1;34;49m"+move[1]+"->"+move[2]+"\033[1;37;49m(\033[1;32;49m"+str(ID)+",\033[1;37;49m"+str(prob[ID])+") ; "
        else:
            moveWithID=move[1]+"->"+move[2]+"("+str(ID)+","+str(prob[ID])+") ; "
        if  ("p" in move): pawnMoves+=moveWithID
        elif("n" in move): knightMoves+=moveWithID
        elif("b" in move): bishopMoves+=moveWithID
        elif("r" in move): rookMoves+=moveWithID
        elif("q" in move): queenMoves+=moveWithID
        elif("k" in move): kingMoves+=moveWithID
        else:
            print("ERROR (functions (createPieceListforPrintOut)): unknown pieceName")
    sortedList.append(pawnMoves)
    sortedList.append(knightMoves)
    sortedList.append(bishopMoves)
    sortedList.append(rookMoves)
    sortedList.append(queenMoves)
    sortedList.append(kingMoves)
    return sortedList

@njit(cache=True)
def printMoves(player,moveList,colored,probNormed,debug=False):
    readableMoveList=[]
    if(debug): print("moveList=",moveList)
    for i,move in enumerate(moveList):
        piece=move[0]
        pieceNew=move[1]
        capturedPiece=move[4]
        castling=move[5]
        enPassant=move[6]
        pos_before=getReadablePosition(player,move[2],debug)
        pos_after=getReadablePosition(player,move[3],debug)
        if(castling[0]==True):
            readableMove=[piece,pos_before,pos_after+"(White CASTLING long)"]
        elif(castling[1]==True):
            readableMove=[piece,pos_before,pos_after+"(Black CASTLING short)"]
        else:
            readableMove=[piece,pos_before,pos_after]
        if(capturedPiece!=""):
            if(enPassant==True):
                readableMove[2]+="(Piece CAPTURED via en-passant)"
            elif(piece=="p" and pieceNew!="p"):
                readableMove[2]+="(Piece CAPTURED + promtion: "+pieceNew+")"
            else:
                readableMove[2]+="(Piece CAPTURED)"
        else:
            if(piece=="p" and pieceNew!="p"):
                readableMove[2]+="(promotion: "+pieceNew+")"
        if(debug): print("DEBUG (functions (printMoves)): readableMove=",readableMove)
        readableMoveList.append(readableMove)
    sortedLists=createPieceListforPrintOut(readableMoveList,probNormed,colored,debug)
    print("INFO (functions (printMoves)): Moves for player",player,":")
    for list in sortedLists:
        print(list)

@njit(cache=True)
def getNNInput(boardpositions, move, debug, noOutputMode):
    isPlayerInCheckBeforeMove=boardpositions.IsPlayerInCheck
    castlingBeforeMove = boardpositions.Castling.copy()
    enPassantBeforeMove = boardpositions.EnPassant
   
    moveInfo,captPiece=boardpositions.playMove(move, False, debug, noOutputMode)
    boardpositions.setIsPlayerInCheck()
    willPlayerBeInCheck=boardpositions.IsPlayerInCheck

    #if("SelfplayNetwork" in boardpositions.GameMode):
    #    nnInput=boardpositions.getInput() #Potential performance boost in this function
    #else:
    nnInput=np.ones(780,dtype=np.float64)
        
    #reverseThePlayedMove
    boardpositions.reverseMove(move, captPiece, castlingBeforeMove, enPassantBeforeMove)
    boardpositions.IsPlayerInCheck=isPlayerInCheckBeforeMove
    boardpositions.Castling=castlingBeforeMove
    boardpositions.EnPassant=enPassantBeforeMove
    return willPlayerBeInCheck,nnInput

@njit(cache=True)
def moveProbs(boardpositions,net,nnInputList,debug):
    lenInput=len(nnInputList)
    if(lenInput<100):
        print("ERROR: More than 100 moves available! Some moved are overlooked!")
    nnInput=np.zeros((100,780),dtype=np.float64)
    for i,inArr in enumerate(nnInputList):
        nnInput[i]=inArr
    if("SelfplayNetwork" in boardpositions.GameMode and boardpositions.CurrentPlayer==0):
        probs=net.feedforward(nnInput[:lenInput])[0]
        #print("probs=",probs)
        moveProbabilities=probs.tolist()
        if(debug): print("Probabilities to win:",moveProbabilities)
    else:
        moveProbabilities=[1.0 for i in range(lenInput)]
    return moveProbabilities

def nextMove(net,boardpositions,colored=False,debug=False,noOutputMode=False):
    #moveStartTime=timer()
    print("test1")
    gameMode=boardpositions.GameMode
    print("test2")
    player=boardpositions.CurrentPlayer
    print("test3")
    chessBoard=boardpositions.ChessBoard
    print("test4")
    finished=False
    print("test5")
    boardpositions.setIsPlayerInCheck()
    print("test6")
    playerInCheck=boardpositions.IsPlayerInCheck
    print("test7")
    if(playerInCheck==True and noOutputMode==False): print("Player",player,"is in CHECK!")
    print("test8")
    outOfCheckMoves,moveProbabilities=boardpositions.findAllowedMoves(net,debug,noOutputMode)
    print("test9")
    if(len(outOfCheckMoves)>0):
        if(len(outOfCheckMoves)<=8):
            #Check if only king is left and opponent can still win!
            if(noOutputMode==False): print("Less than 9 moves available! Check if game is decided.")
            colorPlayer  =boardpositions.Players[boardpositions.CurrentPlayer]
            colorOpponent=boardpositions.Players[boardpositions.CurrentOpponent]
            Knight,Bishop,Rook,Queen,Pawn=colorOpponent+"n",colorOpponent+"b",colorOpponent+"r",colorOpponent+"q",colorOpponent+"p"
            playerPieces,opponentPieces=[],[]
            for row in chessBoard:
                for col in row:
                    if colorPlayer in col:
                        playerPieces.append(col)
                    if colorOpponent in col:
                        opponentPieces.append(col)
            if(noOutputMode==False): print("playerPieces=",playerPieces)
            if(len(playerPieces)==1):
                if(noOutputMode==False): print("opponentPieces=",opponentPieces)
                if(len(opponentPieces)>2 or
                   len(opponentPieces)>1 and (Rook in opponentPieces or Queen in opponentPieces)):# or Pawn in opponentPieces)):
                    for rand,move in enumerate(outOfCheckMoves):
                        if move[4]!="":
                            moveInfoList,captPiece=boardpositions.playMove(move, True, debug, noOutputMode)
                            if(noOutputMode==False): printChessBoard(chessBoard,player,rand,-1.0,colored,moveInfoList,debug)
                            return False
                    boardpositions.setWinner(boardpositions.CurrentOpponent)
                    return True
                elif(len(opponentPieces)==2 and (Bishop in opponentPieces or Knight in opponentPieces)):
                    boardpositions.setWinner(0.5)
                    return True
        maxNum=len(outOfCheckMoves)
        s = np.sum(moveProbabilities)
        probNormed = [float(i)/s for i in moveProbabilities]
        prob_rounded = [round(num, 3) for num in moveProbabilities]
        #probNormed = moveProbabilities/np.sum(moveProbabilities)
        #prob_rounded = np.round(moveProbabilities,3)
        if(gameMode=="WhiteVsBlack" or (gameMode=="WhiteVsComputer" and player==0) or (gameMode=="BlackVsComputer" and player==1)):
            while True:
                try:
                    rand= int(input("Input your move:"))
                    if rand>=maxNum:
                        print("Your integer is too large. Choose between 0 and",maxNum-1)
                        continue
                    else:
                        break
                except ValueError:
                    print("Not an integer!")  
                    continue
            print("Choose move ID",rand)
            move=outOfCheckMoves[rand]
        else:
            if(  gameMode=="SelfplayRandomVsRandom" or (gameMode=="SelfplayNetworkVsRandom" and player==1)):
                if(noOutputMode==False): print("Random mover is moving now!")
                rand=random.randint(0,maxNum-1)
                move=outOfCheckMoves[rand]
            elif(gameMode=="SelfplayNetworkVsNetwork" or (gameMode=="SelfplayNetworkVsRandom" and player==0)):
                if(noOutputMode==False): print("Neural network is moving now!")
                #np.random.choice(np.arange(maxNum), p=probNormed)
                #rand=random.randint(0,maxNum-1)
                rand=np.argmax(moveProbabilities)
                move=outOfCheckMoves[rand]
            else:
                rand=random.randint(0,maxNum)
                move=outOfCheckMoves[rand]
            if(debug): print("(functions (nextMove)): rand=",rand)
        if noOutputMode==False:printMoves(player,outOfCheckMoves,colored,prob_rounded,debug)
        moveInfoList,captPiece=boardpositions.playMove(move, True, debug, noOutputMode)
        if(noOutputMode==False):
            printChessBoard(chessBoard,player,rand,prob_rounded[rand],colored,moveInfoList,debug)
    elif(len(outOfCheckMoves)==0 and playerInCheck==True):
        if(noOutputMode==False): print("Player",player,"is CHECKMATE!")
        boardpositions.setWinner(boardpositions.CurrentOpponent)
        finished=True
    elif(len(outOfCheckMoves)==0 and playerInCheck==False):
        if(noOutputMode==False): print("Player",player,"has no more moves available => Remis.")
        boardpositions.setWinner(0.5)
        finished=True
    #print("time nextMove=",timer()-moveStartTime)
    return finished

@njit(cache=True)
def printChessBoard(chessBoard, player, rand, prob, colored=False, moveInfoList=[], debug=False):
    if(debug): print("moveInfoList=",moveInfoList)
    if(len(moveInfoList)>3): print("ERROR: len(moveInfoList)=",len(moveInfoList),"shouldn't be larger than 3!")
    print("###### Board- ######")
    print("## A B C D E F G H #")
    bkgColor="49"
    for j in range(len(chessBoard)):
        print(8-j,end=' ')
        if(  player==0): colNum=7-j
        elif(player==1): colNum=j
        printColor=""
        for i in range(8):
            if colored:
                if  ((8-j)%2==0 and i%2==0) or ((8-j)%2==1 and i%2==1): bkgColor="44"
                elif((8-j)%2==0 and i%2==1) or ((8-j)%2==1 and i%2==0): bkgColor="46"
                if   "B" in chessBoard[colNum][i]: printColor="\033[0;30;"+bkgColor+"m"
                elif "W" in chessBoard[colNum][i]: printColor="\033[0;37;"+bkgColor+"m"
                else: printColor="\033[1;37;"+bkgColor+"m"
            pieceUnicode=functions.getPieceUnicode(chessBoard[colNum][i],colored)
            print(printColor+pieceUnicode+" ",end='')
        if(colored):
            if(j==2 and len(moveInfoList)>0):
                print("\033[1;37;49m",8-j,"  ",moveInfoList[0],"(\033[1;32;49m"+str(rand)+"\033[1;37;49m,"+str(prob)+")")
            elif(j==4 and len(moveInfoList)>1):
                print("\033[1;37;49m",8-j,"  ",moveInfoList[1])
            elif(j==6 and len(moveInfoList)>2):
                print("\033[1;37;49m",8-j,"  ",moveInfoList[2])
            else:
                print("\033[1;37;49m",8-j)
        else:
            if(j==2 and len(moveInfoList)>0):
                print(8-j,"  ",moveInfoList[0],"("+str(rand)+","+str(prob)+")")
            elif(j==4 and len(moveInfoList)>1):
                print(8-j,"  ",moveInfoList[1])
            elif(j==4 and len(moveInfoList)>2):
                print(8-j,"  ",moveInfoList[2])
            else:
                print(8-j)
    print("# A B C D E F G H ##")
    print("###### -Board ######")

@njit(cache=True)#[types.ListType(types.ListType(types.Array(int64, 1, 'C')))(types.ListType(types.ListType(types.unicode_type)),types.unicode_type)],cache=True)
#     locals={'pPos': types.ListType(types.Array(int64, 1, 'C')),
#             'nPos': types.ListType(types.int64[:]),
#             'bPos': types.ListType(types.int64[:]),
#             'rPos': types.ListType(types.int64[:]),
#             'qPos': types.ListType(types.int64[:]),
#             'kPos': types.ListType(types.int64[:])})
def getPositions(chessBoard,playerSign):
    pPos=typed.List()
    nPos=typed.List()
    bPos=typed.List()
    rPos=typed.List()
    qPos=typed.List()
    kPos=typed.List()
    for row in range(8):
        for col in range(8):
            if   chessBoard[row][col]==playerSign*6: pPos.append( np.array([col,row], dtype=np.int64) )
            elif chessBoard[row][col]==playerSign*5: nPos.append( np.array([col,row], dtype=np.int64) )
            elif chessBoard[row][col]==playerSign*4: bPos.append( np.array([col,row], dtype=np.int64) )
            elif chessBoard[row][col]==playerSign*3: rPos.append( np.array([col,row], dtype=np.int64) )
            elif chessBoard[row][col]==playerSign*2: qPos.append( np.array([col,row], dtype=np.int64) )
            elif chessBoard[row][col]==playerSign*1: kPos.append( np.array([col,row], dtype=np.int64) )
    return typed.List((pPos,nPos,bPos,rPos,qPos,kPos))
