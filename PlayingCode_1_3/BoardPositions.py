import numpy as np
import moves
import functions
from timeit import default_timer as timer
import pickle

class BoardPositions(object):
    def __init__(self):
        self.PlayerColors = ("W","B")
        self.Pieces = ("p","n","b","r","q","k")
        self.NumRows = 8
        self.NumColumns = 8
        self.NumChessField = 64
        self.NumPieces = 6
        self.ChessBoard = np.array([])
        self.Reversed = False
        self.KingPositions = [[4,0],[4,0]]        #From current players view (black's rows are inverted to have a common description)
        self.Castling = [[True,True],[True,True]] #Castling for [[white long,white short],[black long, black short]] still allowed?
        self.EnPassant = [False,[-1,-1]]          #Enpassant allowed in next move; Enpassant[1] is given from opponents view.
        self.Winner = -1
        self.GameMode = ""
        self.CurrentPlayer  =0 # 0=white
        self.CurrentOpponent=1 # 1=black
        self.PlyNumber=1
        self.Finished=False
        self.IsPlayerInCheck=False
        
    def setChessBoard(self, colored=False, debug=False, noOutputMode=False):
        self.ChessBoard=[["  ","  ","  ","  ","  ","  ","  ","  "],
                         ["  ","  ","  ","  ","  ","  ","  ","  "],
                         ["  ","  ","  ","  ","  ","  ","  ","  "],
                         ["  ","  ","  ","  ","  ","  ","  ","  "],
                         ["  ","  ","  ","  ","  ","  ","  ","  "],
                         ["  ","  ","  ","  ","  ","  ","  ","  "],
                         ["  ","  ","  ","  ","  ","  ","  ","  "],
                         ["  ","  ","  ","  ","  ","  ","  ","  "]]
        players=[0,1]
        for player in players:
            playerAbb = self.PlayerColors[player]
            for pieceNumber, pieceName in enumerate(self.Pieces):
                for row in range(self.NumRows):
                    for col in range(self.NumColumns):
                        if(self.PieceBoards[player][pieceNumber][row][col] == 1):
                            self.ChessBoard[row][col] = playerAbb+pieceName
                            if pieceName=="k":
                                if player==self.CurrentPlayer:
                                    field=[col,row]
                                else:
                                    field=[col,7-row]
                                if   player==0: self.KingPositions[0]=field
                                elif player==1: self.KingPositions[1]=field
        if self.CurrentPlayer==1:
            self.Reversed = True
        #if(noOutputMode==False):
        self.printChessBoard(-1,-1.0,colored,[], False)
        print("#### All pieces initialized! ######")
        
    def initializeBoard(self, colored, debug=False, noOutputMode=False):
        self.PieceBoards=np.zeros(2*self.NumPieces*self.NumRows*self.NumColumns)
        self.PieceBoards.shape = (2,self.NumPieces,self.NumRows,self.NumColumns)
        if(debug): print("DEBUG (BoardPositions (definePieceBoards)): All pieceBoards defined:",self.PieceBoards)
        
        self.PieceBoards[0][0][1] = [1,1,1,1,1,1,1,1]                   #white pawns
        self.PieceBoards[0][1][0][1],self.PieceBoards[0][1][0][6] = 1,1 #white knights
        self.PieceBoards[0][2][0][2],self.PieceBoards[0][2][0][5] = 1,1 #white bishops
        self.PieceBoards[0][3][0][0],self.PieceBoards[0][3][0][7] = 1,1 #white rook
        self.PieceBoards[0][4][0][3] = 1 #white queen
        self.PieceBoards[0][5][0][4] = 1 #white king
        self.PieceBoards[1][0][6] = [1,1,1,1,1,1,1,1] #black pawns
        self.PieceBoards[1][1][7][1],self.PieceBoards[1][1][7][6] = 1,1 #black knights
        self.PieceBoards[1][2][7][2],self.PieceBoards[1][2][7][5] = 1,1 #black bishops
        self.PieceBoards[1][3][7][0],self.PieceBoards[1][3][7][7] = 1,1 #black rook
        self.PieceBoards[1][4][7][3] = 1 #black queen
        self.PieceBoards[1][5][7][4] = 1 #black king
        self.setChessBoard(colored)

    def SetStartPosition(self,posFile):
        fileIn = open(posFile,"rb")
        self = pickle.load(fileIn)

    def writePositionToFile(self):
        inputID=self.getInputID(debug)
        fileOut = open("PositionsToPlayFrom/"+inputID+".pkl", "wb")
        pickle.dump([self],fileOut)
        fileOut.close()
        
    def getPositions(self,player,debug=False): #get the positions of pieces [col,row]
        pawnPos,knightPos,bishopPos,rookPos,queenPos,kingPos,playerPos=[],[],[],[],[],[],[]
        colorApp=self.PlayerColors[player]
        for rowNum,row in enumerate(self.ChessBoard):
            for colNum,col in enumerate(row):
                if col==colorApp+"p":  pawnPos.append(  [colNum,rowNum])
                elif col==colorApp+"n":knightPos.append([colNum,rowNum])
                elif col==colorApp+"b":bishopPos.append([colNum,rowNum])
                elif col==colorApp+"r":rookPos.append(  [colNum,rowNum])
                elif col==colorApp+"q":queenPos.append( [colNum,rowNum])
                elif col==colorApp+"k":kingPos.append(  [colNum,rowNum])
        playerPos=[pawnPos,knightPos,bishopPos,rookPos,queenPos,kingPos]
        return playerPos

    def getPieceNumber(self,name):
        pieceNumber=0
        if   name=="p" or name=="P": pieceNumber=0
        elif name=="n" or name=="N": pieceNumber=1
        elif name=="b" or name=="B": pieceNumber=2
        elif name=="r" or name=="R": pieceNumber=3
        elif name=="q" or name=="Q": pieceNumber=4
        elif name=="k" or name=="K": pieceNumber=5
        else:print("ERROR (BoardPositions (getPieceNumber)): PieceName",name,"unknown!")
        return pieceNumber

    def getNormalMoves(self,ownPositions,debug=False, noOutputMode=False):
        normalMoves,normalNNinp=[],[]
        onlyCaptureMoves=False
        moveID=0
        for pieceNum,piece in enumerate(self.Pieces):
            if(debug): print("DEBUG (BoardPositions (getNormalMoves)): Current piece=",piece,"; ownPositions=",ownPositions[pieceNum])
            for piecePos_i in ownPositions[pieceNum]:
                delta=moves.getBasicMoves(piece,piecePos_i,onlyCaptureMoves,debug)
                if(debug): print("DEBUG (moves (getAllowedMovesForPiece)): piece=",piece)
                for Delta in delta:
                    validMove,capturedPiece=True,""
                    i=1
                    while validMove==True and capturedPiece=="":
                        Del=[d*i for d in Delta]
                        if(debug): print("DEBUG (moves (getNormalMoves)): Del=",Del)
                        newPos,validMove,capturedPiece=moves.checkMove(self.CurrentPlayer,piece,piecePos_i,Del,self.ChessBoard,debug)
                        if(validMove==True):
                            if(piece=="p" and piecePos_i[1]==6 and newPos[1]==7):
                                ####  [piece before,piece after,pos before,pos after,capturedPiece,castle?      ,enPassant?,moveID]
                                move1=[piece       ,"n"        ,piecePos_i,newPos   ,capturedPiece,[False,False],False     ,moveID]
                                move2=[piece       ,"b"        ,piecePos_i,newPos   ,capturedPiece,[False,False],False     ,moveID]
                                move3=[piece       ,"r"        ,piecePos_i,newPos   ,capturedPiece,[False,False],False     ,moveID]
                                move4=[piece       ,"q"        ,piecePos_i,newPos   ,capturedPiece,[False,False],False     ,moveID]
                                willPlayerBeInCheck1,nnIn1=functions.getNNInput(self, move1, debug, noOutputMode)
                                willPlayerBeInCheck2,nnIn2=functions.getNNInput(self, move2, debug, noOutputMode)
                                willPlayerBeInCheck3,nnIn3=functions.getNNInput(self, move3, debug, noOutputMode)
                                willPlayerBeInCheck4,nnIn4=functions.getNNInput(self, move4, debug, noOutputMode)
                                if(willPlayerBeInCheck1==False):
                                    normalMoves.append(move1)
                                    normalNNinp.append(nnIn1)
                                    moveID+=1
                                    normalMoves.append(move2)
                                    normalNNinp.append(nnIn2)
                                    moveID+=1
                                    normalMoves.append(move3)
                                    normalNNinp.append(nnIn3)
                                    moveID+=1
                                    normalMoves.append(move4)
                                    normalNNinp.append(nnIn4)
                                    moveID+=1
                            else:
                                move=[piece,piece,piecePos_i,newPos,capturedPiece,[False,False],False,moveID]
                                willPlayerBeInCheck,nnInp=functions.getNNInput(self, move, debug, noOutputMode)
                                if(willPlayerBeInCheck==False):
                                    normalMoves.append(move)
                                    normalNNinp.append(nnInp)
                                    moveID+=1
                            if(piece=="p" or piece=="n" or piece=="k"):
                                break
                            else:
                                i+=1
        if(debug):
            print("DEBUG (moves (getNormalMoves)): normalMoves=",normalMoves)
        return normalMoves,normalNNinp

    def getCastlingMoves(self,moveID,debug,noOutputMode=False):
        castlingMoves,castlingNNinp=[],[]
        kingPosition=self.KingPositions[self.CurrentPlayer]
        chessBoard=self.ChessBoard
        if(kingPosition==[4,0]):
            if(self.Castling[0][0]==True and chessBoard[0][1]=="  " and chessBoard[0][2]=="  " and chessBoard[0][3]=="  "): #castling long
                if(debug): print("DEBUG (BoardPositions (getCastlingMoves)): Adding castling LONG to player",self.CurrentPlayer,"'s castlingMoves.")
                move=         ["k","k",[4,0],[2,0],"",[True,False],False,moveID]
                inBetweenMove=["k","k",[4,0],[3,0],"",[True,False],False,moveID]
                willPlayerBeInCheck ,nnInp=functions.getNNInput(self, move         , debug, noOutputMode)
                willPlayerBeInCheck1,nnIn1=functions.getNNInput(self, inBetweenMove, debug, noOutputMode)
                if(willPlayerBeInCheck==False and willPlayerBeInCheck1==False):
                    castlingMoves.append(move)
                    castlingNNinp.append(nnInp)
                    moveID+=1
            if(self.Castling[0][1]==True and chessBoard[0][6]=="  " and chessBoard[0][5]=="  "): #castling short
                if(debug): print("DEBUG (BoardPositions (getCastlingMoves)): Adding castling SHORT to player",self.CurrentPlayer,"'s castlingMoves.")
                move         =["k","k",[4,0],[6,0],"",[False,True],False,moveID]
                inBetweenMove=["k","k",[4,0],[5,0],"",[False,True],False,moveID]
                willPlayerBeInCheck ,nnInp=functions.getNNInput(self, move         , debug, noOutputMode)
                willPlayerBeInCheck1,nnIn1=functions.getNNInput(self, inBetweenMove, debug, noOutputMode)
                if(willPlayerBeInCheck==False and willPlayerBeInCheck1==False):
                    castlingMoves.append(move)
                    castlingNNinp.append(nnInp)
                    moveID+=1
        return castlingMoves,castlingNNinp

    def getEnPassantMoves(self,moveID,pawnPositions,debug=False,noOutputMode=False):
        enPassantMoves,enPassantNNinp=[],[]
        oppPawnOppPerspective=self.EnPassant[1]
        oppPawn=(oppPawnOppPerspective[0],7-oppPawnOppPerspective[1])
        for pawn in pawnPositions:
            if(pawn[1]==4 and oppPawn[1]==4 and (pawn[0]==oppPawn[0]-1 or pawn[0]==oppPawn[0]+1)):
                if(debug): print("INFO: En-passant move available for player",self.CurrentPlayer,"'s pawn at",pawn)
                position_after=(oppPawn[0],oppPawn[1]+1)
                move=["p","p",pawn,position_after,"p",[False,False],True,moveID]
                willPlayerBeInCheck,nnInp=functions.getNNInput(self, move, debug, noOutputMode)
                if(willPlayerBeInCheck==False):
                    enPassantMoves.append(move)
                    enPassantNNinp.append(nnInp)
                    moveID+=1
        return enPassantMoves,enPassantNNinp
    
    def findAllowedMoves(self,Sizes,Weights,Biases,debug=False,noOutputMode=False):
        ownPositions=self.getPositions(self.CurrentPlayer)
        oppPositions=self.getPositions(self.CurrentOpponent)
        if(debug):
            print("DEBUG (BoardPositions (findAllowedMoves)): Own piece positions of player:",self.CurrentPlayer  ,":",ownPositions)
            print("DEBUG (BoardPositions (findAllowedMoves)): Opp piece positions of player:",self.CurrentOpponent,":",oppPositions)
        castlingMoves,castlingNNinp,enPassantMoves,enPassantNNinp=[],[],[],[]
        normalMoves,normalNNinp=self.getNormalMoves(ownPositions,debug,noOutputMode)
        if(self.IsPlayerInCheck==False and (self.Castling[0][0]==True or self.Castling[0][1]==True)):
            castlingMoves,castlingNNinp=self.getCastlingMoves(len(normalMoves),debug,noOutputMode)
        if(self.EnPassant[0]==True):
            enPassantMoves,enPassantNNinp=self.getEnPassantMoves(len(normalMoves)+len(castlingMoves),ownPositions[0],debug,noOutputMode)
        if(debug):
            print("DEBUG (BoardPositions (findAllowedMoves)): normalMoves="   ,normalMoves)
            print("DEBUG (BoardPositions (findAllowedMoves)): castlingMoves=" ,castlingMoves)
            print("DEBUG (BoardPositions (findAllowedMoves)): enPassantMoves=",enPassantMoves)
        outOfCheckMoves=normalMoves+castlingMoves+enPassantMoves
        outOfCheckNNList=normalNNinp+castlingNNinp+enPassantNNinp
        if len(outOfCheckMoves)>0:
            outOfCheckNNinp=np.vstack(normalNNinp+castlingNNinp+enPassantNNinp)
            moveProbabilites=functions.moveProbs(self,Sizes,Weights,Biases,outOfCheckNNinp, debug)
        else:
            moveProbabilites=[]
        return outOfCheckMoves,moveProbabilites

    def nextPly(self):
        # Reverse the board
        if(  self.Reversed==True ): self.Reversed=False
        elif(self.Reversed==False): self.Reversed=True
        self.ChessBoard=np.flip(self.ChessBoard,0)
        for playerNum,player in enumerate(self.PieceBoards):
            for pieceNum,piece in enumerate(player):
                self.PieceBoards[playerNum][pieceNum]=np.flip(self.PieceBoards[playerNum][pieceNum],0)
        # Reverse the castling properties
        helperCastling=self.Castling[0]
        self.Castling[0]=self.Castling[1]
        self.Castling[1]=helperCastling
        # Reverse player <-> opponent
        helperPlayer=self.CurrentOpponent
        self.CurrentOpponent=self.CurrentPlayer
        self.CurrentPlayer=helperPlayer
        # Increas ply number
        self.PlyNumber+=1
        
    def printChessBoard(self, rand, prob, colored=False, moveInfoList=[], debug=False):
        if(debug): print("moveInfoList=",moveInfoList)
        if(len(moveInfoList)>3): print("ERROR: len(moveInfoList)=",len(moveInfoList),"shouldn't be larger than 3!")
        print("###### Board- ######")
        print("## A B C D E F G H #")
        bkgColor="49"
        for j in range(len(self.ChessBoard)):
            print(8-j,end=' ')
            if(  self.CurrentPlayer==0): colNum=7-j
            elif(self.CurrentPlayer==1): colNum=j
            printColor=""
            for i in range(8):
                if colored:
                    if  ((8-j)%2==0 and i%2==0) or ((8-j)%2==1 and i%2==1): bkgColor="44"
                    elif((8-j)%2==0 and i%2==1) or ((8-j)%2==1 and i%2==0): bkgColor="46"
                    if   "B" in self.ChessBoard[colNum][i]: printColor="\033[0;30;"+bkgColor+"m"
                    elif "W" in self.ChessBoard[colNum][i]: printColor="\033[0;37;"+bkgColor+"m"
                    else: printColor="\033[1;37;"+bkgColor+"m"
                pieceUnicode=functions.getPieceUnicode(self.ChessBoard[colNum][i],colored)
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
                            
    def playMove(self,move, writeMoveInfoList=False, debug=False, noOutputMode=False):
        player,opponent=self.CurrentPlayer,self.CurrentOpponent
        piece ,pieceNew=move[0],move[1]
        oldPos,newPos  =move[2],move[3]
        capturedPiece=move[4]
        isCastling,isEnPassant=move[5],move[6]
        returnStringList=[]
        if(writeMoveInfoList):
            oldPosReadable=functions.getReadablePosition(player,oldPos,debug)
            newPosReadable=functions.getReadablePosition(player,newPos,debug)
            returnStringList.append("Player "+str(player)+": "+piece+" from "+oldPosReadable+" to "+newPosReadable)
        if  (capturedPiece!="" and isEnPassant==True):
            if(noOutputMode==False and debug): print("INFO: Player",player,"just captured via en-passant at",oldPos)
            oppPawn=[newPos[0],newPos[1]-1]
            self.removePiece(opponent, "p", oppPawn, False)
            capturedPiece="p"
            if(writeMoveInfoList):
                returnStringList.append("Player "+str(player)+" captured "+capturedPiece+" via en-passant")
        elif(capturedPiece!="" and isEnPassant==False):
            if(writeMoveInfoList):
                returnStringList.append("Player "+str(player)+" captured "+capturedPiece+" at "+newPosReadable)
            for pieceOpp in self.Pieces:
                self.removePiece(opponent, pieceOpp, newPos, False)
        elif(capturedPiece=="" and isEnPassant==True):
            print("ERROR: En-passant is always capture move!")
        self.removePiece(player, piece, oldPos, False)
        self.setPiece(player, pieceNew, newPos, False)
        if(pieceNew!=piece):
            if(piece=="p"):
                if(writeMoveInfoList): returnStringList.append("Player "+str(player)+"'s promoted his pawn to a "+pieceNew+" at "+newPosReadable)
            else:
                print("ERROR: Only pawns can be promoted.")
        if  (isCastling[0]==True): # castling long
            if(noOutputMode==False and debug): print("Player",player,"is CASTLING long!")
            self.removePiece(player, "r", [0,0], False)
            self.setPiece(   player, "r", [3,0], False)
            self.Castling[0]=[False,False]
            if(writeMoveInfoList): returnStringList.append("Player "+str(player)+" castled long.")
            if(isEnPassant==True):print("ERROR: Impossible to be en-passant and castle move at the same time!")
        elif(isCastling[1]==True): # castling short
            if(noOutputMode==False and debug): print("Player",player,"is CASTLING short!")
            self.removePiece(player, "r", [7,0], False)
            self.setPiece(   player, "r", [5,0], False)
            self.Castling[0]=[False,False]
            if(writeMoveInfoList): returnStringList.append("Player "+str(player)+" castled short.")
            if(isEnPassant==True):print("ERROR: Impossible to be en-passant and castle move at the same time!")
        if(self.Castling[0][0]==True or self.Castling[0][1]==True):
            if(piece=="k" and oldPos==[4,0]):
                if(noOutputMode==False and writeMoveInfoList==True):
                    print("INFO: Player",player,"is no longer allowed to castle.")
                self.Castling[0]=[False,False]
            elif(piece=="r" and oldPos==[0,0]):
                if(self.Castling[0][0]==True):
                    if(noOutputMode==False and writeMoveInfoList==True):
                        print("INFO: Player",player,"is no longer allowed to castle long.")
                self.Castling[0][0]=False
            elif(piece=="r" and oldPos==[7,0]):
                if(self.Castling[0][1]==True):
                    if(noOutputMode==False and writeMoveInfoList==True):
                        print("INFO: Player",player,"is no longer allowed to castle short.")
                self.Castling[0][1]=False
        if(piece=="p" and oldPos[1]==1 and newPos[1]==3):
            self.EnPassant=[True,newPos]
        else:
            self.EnPassant=[False,[-1,-1]]
        return returnStringList,capturedPiece

    def reverseMove(self,move,capturedPiece,castlingBeforeMove,enPassantBeforeMove):
        player,opponent=self.CurrentPlayer,self.CurrentOpponent
        piece ,pieceNew=move[0],move[1]
        oldPos,newPos  =move[2],move[3]
        capturedPiece=move[4]
        isCastling,isEnPassant=move[5],move[6]
        self.removePiece(player, pieceNew, newPos, False)
        self.setPiece(player, piece, oldPos, False)
        if  (capturedPiece!="" and isEnPassant==True):
            oppPawn=[newPos[0],newPos[1]-1]
            self.setPiece(opponent, "p", oppPawn, False)
        elif(capturedPiece!="" and isEnPassant==False):
            self.setPiece(opponent, capturedPiece[1:], newPos, False)
        if  (isCastling[0]==True): # castling long
            self.setPiece(   player, "r", [0,0], False)
            self.removePiece(player, "r", [3,0], False)
            self.Castling=castlingBeforeMove
        elif(isCastling[1]==True): # castling short
            self.setPiece(   player, "r", [7,0], False)
            self.removePiece(player, "r", [5,0], False)
            self.Castling=castlingBeforeMove
        self.EnPassant=enPassantBeforeMove

    def setIsPlayerInCheck(self, debug=False):
        playerIsInCheck=False
        kingPosition=self.KingPositions[self.CurrentPlayer]
        ownColor,oppColor=self.PlayerColors[self.CurrentPlayer],self.PlayerColors[self.CurrentOpponent]
        #print("kingPosition,ownColor,oppColor=",kingPosition,ownColor,oppColor)
        pos=[0,0]
        basicMoves=(((1,1),(-1,1),(1,-1),(-1,-1)), #bishopMoves
                    ((1,0),(-1,0),(0, 1), (0,-1)), #rookMoves
                    ((1,2),(-1,2),(1,-2),(-1,-2),(2,1),(-2,1),(2,-1),(-2,-1)), #knightMoves
                    ((1,1),(-1,1))) #pawnMoves 
        for num,moves in enumerate(basicMoves):
            for move in moves:
                i=1
                while i<=8:
                    pos[0]=kingPosition[0]+move[0]*i
                    pos[1]=kingPosition[1]+move[1]*i
                    if(pos[0]<0 or pos[0]>7 or pos[1]<0 or pos[1]>7 or (ownColor in self.ChessBoard[pos[1]][pos[0]])):
                        break
                    elif(oppColor in self.ChessBoard[pos[1]][pos[0]]):
                        oppQ,oppB,oppR,oppN,oppP,oppK=oppColor+"q",oppColor+"b",oppColor+"r",oppColor+"n",oppColor+"p",oppColor+"k"
                        if((num==0 and (self.ChessBoard[pos[1]][pos[0]]==oppQ or self.ChessBoard[pos[1]][pos[0]]==oppB or (i==1 and self.ChessBoard[pos[1]][pos[0]]==oppK))) or
                           (num==1 and (self.ChessBoard[pos[1]][pos[0]]==oppQ or self.ChessBoard[pos[1]][pos[0]]==oppR or (i==1 and self.ChessBoard[pos[1]][pos[0]]==oppK))) or
                           (num==2 and  self.ChessBoard[pos[1]][pos[0]]==oppN) or
                           (num==3 and  self.ChessBoard[pos[1]][pos[0]]==oppP)):
                            playerIsInCheck=True
                        elif((num==0 or num==1) and oppColor in self.ChessBoard[pos[1]][pos[0]]):
                            break
                    if(num==2 or num==3):
                        break
                    i+=1
        self.IsPlayerInCheck=playerIsInCheck
                    
    def removePiece(self, player, piece, oldPos, debug=False):
        pieceNumber=self.getPieceNumber(piece)
        self.PieceBoards[player][pieceNumber][oldPos[1]][oldPos[0]]=0
        self.ChessBoard[oldPos[1]][oldPos[0]]="  "

    def setPiece(self, player, piece, newPos, debug=False):
        pieceNumber=self.getPieceNumber(piece)
        playerAbb=self.PlayerColors[player]
        self.PieceBoards[player][pieceNumber][newPos[1]][newPos[0]]=1
        self.ChessBoard[newPos[1]][newPos[0]]=playerAbb+piece
        if(piece=="k"):
            self.KingPositions[player]=newPos

    def setWinner(self,winner):
        self.Winner=winner

    def getWinner(self):
        return self.Winner

    def getInput(self):
        pB=self.PieceBoards
        #print("pB=",pB)
        if(self.CurrentPlayer==0):
            flattenedInput=np.vstack((pB[0][0].ravel(),pB[0][1].ravel(),pB[0][2].ravel(),pB[0][3].ravel(),pB[0][4].ravel(),pB[0][5].ravel(),\
                                      pB[1][0].ravel(),pB[1][1].ravel(),pB[1][2].ravel(),pB[1][3].ravel(),pB[1][4].ravel(),pB[1][5].ravel())).ravel()
        elif(self.CurrentPlayer==1):
            flattenedInput=np.vstack((pB[1][0].ravel(),pB[1][1].ravel(),pB[1][2].ravel(),pB[1][3].ravel(),pB[1][4].ravel(),pB[1][5].ravel(),\
                                      pB[0][0].ravel(),pB[0][1].ravel(),pB[0][2].ravel(),pB[0][3].ravel(),pB[0][4].ravel(),pB[0][5].ravel())).ravel()
        else:
            print("ERROR (BoardPositions (getInput): Wrong player input=",self.CurrentPlayer)
        castlingArray=np.zeros(4)
        enpassantArray=np.zeros(8)
        if(self.Castling[0][0]==True): castlingArray[0]=1
        if(self.Castling[0][1]==True): castlingArray[1]=1
        if(self.Castling[1][0]==True): castlingArray[2]=1
        if(self.Castling[1][1]==True): castlingArray[3]=1
        if(self.EnPassant[0]==True):   enpassantArray[self.EnPassant[1][0]]=1
        flattenedInput=np.concatenate((flattenedInput,enpassantArray,castlingArray))
        #print("flattenedInput=",flattenedInput)
        return flattenedInput

    def reconstructGameState(self,flattenedInput,plyNumber):
        if plyNumber%2==1:
            self.CurrentPlayer,self.CurrentOpponent=0,1
            self.Reversed = False
        elif plyNumber%2==0:
            self.CurrentPlayer,self.CurrentOpponent=1,0
            self.Reversed = True
        self.PlyNumber=plyNumber
        self.PieceBoards=flattenedInput[0:768]
        self.PieceBoards.shape = (2,self.NumPieces,self.NumRows,self.NumColumns)
        if self.CurrentPlayer==1:
            self.PieceBoards=np.flip(self.PieceBoards,0)
        self.setChessBoard(False, False, False)
        enPassantArray=flattenedInput[768:776]
        print("enPassantArray=",enPassantArray)
        self.EnPassant = [False,[-1,-1]]        
        for i,elem in enumerate(enPassantArray):
            if elem==1:
                self.EnPassant[0]=True
                self.EnPassant[1]=[i,3]
        castlingArray =flattenedInput[776:780]
        print("castlingArray=",castlingArray)
        print("self.EnPassant=",self.EnPassant)
        self.Castling=[[True,True],[True,True]]
        if(castlingArray[0]==0): self.Castling[0][0]=False
        if(castlingArray[1]==0): self.Castling[0][1]=False
        if(castlingArray[2]==0): self.Castling[1][0]=False
        if(castlingArray[3]==0): self.Castling[1][1]=False

    def getInputID(self,debug):
        if(debug): print("ChessBoard=",self.ChessBoard)
        playerAbb=self.PlayerColors[self.CurrentPlayer]
        oppAbb   =self.PlayerColors[self.CurrentOpponent]
        InputIDString=""
        for rowNum,row in enumerate(self.ChessBoard):
            countEmpty=0
            for colNum,col in enumerate(row):
                if countEmpty==0:
                    num=""
                else:
                    num=str(countEmpty)
                if(playerAbb in col):
                    #stringToAppend=str(colNum)+col[-1:]
                    InputIDString+=num+col[-1:]
                    countEmpty=0
                elif(oppAbb in col):
                    #stringToAppend=str(colNum)+col[-1:].upper()
                    InputIDString+=num+col[-1:].upper()
                    countEmpty=0
                else:
                    countEmpty+=1
            if countEmpty>0 and countEmpty<8:
                InputIDString+=str(countEmpty)
            if(rowNum<7):
                InputIDString+="/"
        if(self.Castling[0][0]==True or self.Castling[0][1]==True or self.Castling[1][0]==True or self.Castling[1][1]==True):
            InputIDString+="_"
            if(self.Castling[0][0]==True): InputIDString+="l"
            if(self.Castling[0][1]==True): InputIDString+="s"
            if(self.Castling[1][0]==True): InputIDString+="L"
            if(self.Castling[1][1]==True): InputIDString+="S"
        if(self.EnPassant[0]==True):
            xpos_plus=self.EnPassant[1][0]+1
            xpos_minus=self.EnPassant[1][0]-1
            if((xpos_plus<=7 and self.ChessBoard[4][xpos_plus]==playerAbb+"p") or (xpos_minus>=0 and self.ChessBoard[4][xpos_minus]==playerAbb+"p")):
                InputIDString+="_"+str(self.EnPassant[1][0])
        if(debug): print("InputIDString =",InputIDString)
        return InputIDString

    def setGameMode(self,gameMode):
        self.GameMode=gameMode
