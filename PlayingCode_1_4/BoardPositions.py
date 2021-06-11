import numpy as np
import moves
import functions
from timeit import default_timer as timer
import pickle
import numba as nb
from numba.experimental import jitclass

"""
@nb.njita
def flipud(x):
    return x[::-1]
"""
spec = [
    ('Players'         , nb.types.ListType(nb.types.int64)), # +1, -1
    ('Pieces'          , nb.types.ListType(nb.types.int64)),
    ('KingPositions'   , nb.types.Array(nb.types.int64, 2, 'C')), #nb.int64[:,:]),
    ('Castling'        , nb.types.Array(nb.types.int64, 2, 'C')), #nb.int64[:,:]),
    ('EnPassant'       , nb.types.int64),
    ('Winner'          , nb.int64),
    ('GameMode'        , nb.types.string),
    ('CurrentPlayer'   , nb.int64),
    ('CurrentOpponent' , nb.int64),
    ('PlyNumber'       , nb.int64),
    ('Reversed'        , nb.boolean),
    ('Finished'        , nb.boolean),
    ('IsPlayerInCheck' , nb.boolean),
    ('ChessBoard'      , nb.types.Array(nb.types.int64, 2, 'C')),
    ('PieceBoards'     , nb.types.Array(nb.types.int64, 4, 'C')), #nb.int64[:,:,:,:]),
]
@jitclass(spec)
class BoardPositions(object):
    def __init__(self):
        self.Players=nb.typed.List((+1,-1))        #+1: W, -1:B
        self.Pieces = nb.typed.List((1,2,3,4,5,6)) #1: k, 2:q, 3:r, 4:b, 5:n, 6:p
        self.KingPositions = np.array([[4,0],[4,0]], dtype=np.int64) #From current players view (black's rows are inverted to have a common description)
        self.Castling = np.array([[1,1],[1,1]], dtype=np.int64)      #[[1=True,1=True],[1=True,1=True]] #Castling for [[white long,white short],[black long, black short]] still allowed?
        self.EnPassant = -1                                          #Enpassant colum. If -1 en-passant is not allowed in next move
        self.Winner = -1
        self.GameMode = ""
        self.CurrentPlayer  =+1 # +1=white
        self.CurrentOpponent=-1 # -1=black
        self.PlyNumber=1
        self.Reversed = False
        self.Finished=False
        self.IsPlayerInCheck=False
        
    def setChessBoard(self,idx):
        self.ChessBoard=np.zeros((8,8), dtype=np.int64)
        for playerSign in self.Players:
            playerNumerator=functions.getPlayerNumerator(playerSign)
            for piece in self.Pieces:
                pieceNumerator=piece-1
                for row in range(8):
                    for col in range(8):
                        if(self.PieceBoards[playerNumerator][pieceNumerator][row][col] == 1):
                            self.ChessBoard[row][col] = playerSign*piece
                            if piece==1:
                                if playerSign==self.CurrentPlayer:
                                    self.KingPositions[0]=[col,row]
                                else:
                                    self.KingPositions[1]=[col,7-row]
        if self.CurrentPlayer==-1:
            self.Reversed = True
        #functions.printChessBoard(self.ChessBoard,self.CurrentPlayer,-1,-1.0)
        if(idx==0):
            print("###### Board- ######")
            print("## A B C D E F G H #")
            for j in range(len(self.ChessBoard)):
                ChessBoardString=str(8-j)+" "
                colNum=j
                if self.CurrentPlayer==1: colNum=7-j
                printColor=""
                for i in range(8):
                    pieceUnicode=functions.getPieceUnicode(self.ChessBoard[colNum][i],False)
                    ChessBoardString+=printColor+pieceUnicode+" "
                ChessBoardString+=" "+str(8-j)
                print(ChessBoardString)
            print("# A B C D E F G H ##")
            print("###### -Board ######\n")
        
    def initializeBoard(self, idx, colored, debug=False, noOutputMode=False):
        self.PieceBoards=np.zeros((2,6,8,8), dtype=np.int64)
        self.PieceBoards[0][5][1] = [1,1,1,1,1,1,1,1]                   #white pawns
        self.PieceBoards[0][4][0][1],self.PieceBoards[0][4][0][6] = 1,1 #white knights
        self.PieceBoards[0][3][0][2],self.PieceBoards[0][3][0][5] = 1,1 #white bishops
        self.PieceBoards[0][2][0][0],self.PieceBoards[0][2][0][7] = 1,1 #white rook
        self.PieceBoards[0][1][0][3] = 1                                #white queen
        self.PieceBoards[0][0][0][4] = 1                                #white king
        self.PieceBoards[1][5][6] = [1,1,1,1,1,1,1,1]                   #black pawns
        self.PieceBoards[1][4][7][1],self.PieceBoards[1][4][7][6] = 1,1 #black knights
        self.PieceBoards[1][3][7][2],self.PieceBoards[1][3][7][5] = 1,1 #black bishops
        self.PieceBoards[1][2][7][0],self.PieceBoards[1][2][7][7] = 1,1 #black rook
        self.PieceBoards[1][1][7][3] = 1                                #black queen
        self.PieceBoards[1][0][7][4] = 1                                #black king
        self.setChessBoard(idx)
        if(debug): print("DEBUG (BoardPositions (definePieceBoards)): All pieceBoards initialized:",self.PieceBoards)
        
    def SetStartPosition(self,posFile):
        fileIn = open(posFile,"rb")
        self = pickle.load(fileIn)

    def writePositionToFile(self):
        inputID=self.getInputID(debug)
        fileOut = open("PositionsToPlayFrom/"+inputID+".pkl", "wb")
        pickle.dump([self],fileOut)
        fileOut.close()
    
    def getNormalMoves(self,ownPositions,debug=False, noOutputMode=False):
        normalMoves=nb.typed.List()
        normalNNinp=nb.typed.List()
        onlyCaptureMoves=False
        moveID=0
        for piece in self.Pieces:
            pieceNum=piece-1
            if(debug): print("DEBUG (BoardPositions (getNormalMoves)): Current piece=",piece,"; ownPositions=",ownPositions[pieceNum])
            for piecePos_i in ownPositions[pieceNum]:
                delta=moves.getBasicMoves(piece,piecePos_i,onlyCaptureMoves,debug)
                if(debug): print("DEBUG (BoardPositions (getNormalMoves)): piece=",functions.getPieceName(piece))
                for Delta in delta:
                    validMove,capturedPiece=True,0
                    i=1
                    while validMove==True and capturedPiece==0:
                        Del=[d*i for d in Delta]
                        if(debug): print("DEBUG (BoardPositions (getNormalMoves)): Del=",Del)
                        newPos,validMove,capturedPiece=moves.checkMove(self.CurrentPlayer,piece,piecePos_i,Del,self.ChessBoard,debug)
                        if(validMove==True):
                            if(piece==6 and piecePos_i[1]==6 and newPos[1]==7):
                                ####  [ piece ,           pos before        ,       pos after     ,  captured?    , castle?,enPassant?,moveID]
                                move1=[piece,5,  piecePos_i[0],piecePos_i[1],  newPos[0],newPos[1],  capturedPiece,  0,0,      0      ,moveID]
                                move2=[piece,4,  piecePos_i[0],piecePos_i[1],  newPos[0],newPos[1],  capturedPiece,  0,0,      0      ,moveID]
                                move3=[piece,3,  piecePos_i[0],piecePos_i[1],  newPos[0],newPos[1],  capturedPiece,  0,0,      0      ,moveID]
                                move4=[piece,2,  piecePos_i[0],piecePos_i[1],  newPos[0],newPos[1],  capturedPiece,  0,0,      0      ,moveID]
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
                                move=[piece,piece,  piecePos_i[0],piecePos_i[1],  newPos[0],newPos[1],  capturedPiece,  0,0,  0  ,moveID]
                                willPlayerBeInCheck,nnInp=functions.getNNInput(self, move, debug, noOutputMode)
                                if(willPlayerBeInCheck==False):
                                    normalMoves.append(move)
                                    normalNNinp.append(nnInp)
                                    moveID+=1
                            if(piece==6 or piece==5 or piece==1): #(piece=="p" or piece=="n" or piece=="k"):
                                break
                            else:
                                i+=1
        return normalMoves,normalNNinp

    def getCastlingMoves(self,moveID,debug,noOutputMode=False):
        castlingMoves=nb.typed.List()#[[np.int64(x),np.int64(x),np.int64(x),np.int64(x),np.int64(x),np.int64(x),np.int64(x),np.int64(x),np.int64(x),np.int64(x)] for x in range(0)])
        castlingNNinp=nb.typed.List()#[np.ones(780,dtype=np.float64) for x in range(0)])
        kingPosition=self.KingPositions[0]
        chessBoard=self.ChessBoard
        if(kingPosition[0]==4 and kingPosition[1]==0):
            if(self.Castling[0][0]==1 and chessBoard[0][1]==0 and chessBoard[0][2]==0 and chessBoard[0][3]==0): #castling long
                if(debug): print("DEBUG (BoardPositions (getCastlingMoves)): Adding castling LONG to player",self.CurrentPlayer,"'s castlingMoves.")
                move         = [1,1,  4,0,  2,0,  0,  1,0,  0  ,moveID]
                inBetweenMove= [1,1,  4,0,  3,0,  0,  1,0,  0  ,moveID]
                willPlayerBeInCheck ,nnInp=functions.getNNInput(self, move         , debug, noOutputMode)
                willPlayerBeInCheck1,nnIn1=functions.getNNInput(self, inBetweenMove, debug, noOutputMode)
                if(willPlayerBeInCheck==False and willPlayerBeInCheck1==False):
                    castlingMoves.append(move)
                    castlingNNinp.append(nnInp)
                    moveID+=1
            if(self.Castling[0][1]==1 and chessBoard[0][6]==0 and chessBoard[0][5]==0): #castling short
                if(debug): print("DEBUG (BoardPositions (getCastlingMoves)): Adding castling SHORT to player",self.CurrentPlayer,"'s castlingMoves.")
                move         = [1,1,  4,0,  6,0,  0,  0,1,  0  ,moveID]
                inBetweenMove= [1,1,  4,0,  5,0,  0,  0,1,  0  ,moveID]
                willPlayerBeInCheck ,nnInp=functions.getNNInput(self, move         , debug, noOutputMode)
                willPlayerBeInCheck1,nnIn1=functions.getNNInput(self, inBetweenMove, debug, noOutputMode)
                if(willPlayerBeInCheck==False and willPlayerBeInCheck1==False):
                    castlingMoves.append(move)
                    castlingNNinp.append(nnInp)
                    moveID+=1
        return castlingMoves,castlingNNinp

    def getEnPassantMoves(self,moveID,pawnPositions,debug=False,noOutputMode=False):
        enPassantMoves=nb.typed.List()#[[np.int64(x),np.int64(x),np.int64(x),np.int64(x),np.int64(x),np.int64(x),np.int64(x),np.int64(x),np.int64(x),np.int64(x)] for x in range(0)])
        enPassantNNinp=nb.typed.List()#[np.ones(780,dtype=np.float64) for x in range(0)])
        for pawn in pawnPositions:
            if(pawn[1]==4 and (pawn[0]==self.EnPassant-1 or pawn[0]==self.EnPassant+1)):
                if(debug): print("INFO: En-passant move available for player",self.CurrentPlayer,"'s pawn at",pawn)
                position_after=[self.EnPassant,5]
                move=[6,6,  pawn[0],pawn[1],  position_after[0],position_after[1],  6,  0,0,  1  ,moveID]
                willPlayerBeInCheck,nnInp=functions.getNNInput(self, move, debug, noOutputMode)
                if(willPlayerBeInCheck==False):
                    enPassantMoves.append(move)
                    enPassantNNinp.append(nnInp)
                    moveID+=1
        return enPassantMoves,enPassantNNinp
    
    def findAllowedMoves(self,Sizes,Weights,Biases,debug=False,noOutputMode=False):
        ownPositions=functions.getPositions(self.ChessBoard,self.CurrentPlayer)
        oppPositions=functions.getPositions(self.ChessBoard,self.CurrentOpponent)
        if(debug):
            print("DEBUG (BoardPositions (findAllowedMoves)): Own piece positions of player:",self.CurrentPlayer  ,":",ownPositions)
            print("DEBUG (BoardPositions (findAllowedMoves)): Opp piece positions of player:",self.CurrentOpponent,":",oppPositions)
        outOfCheckMoves,outOfCheckNNinp=self.getNormalMoves(ownPositions,debug,noOutputMode)
        if(debug): print("DEBUG (BoardPositions (findAllowedMoves)): normalMoves=",outOfCheckMoves)
        if noOutputMode==False: print("self.Castling=",self.Castling)
        if(self.IsPlayerInCheck==False and (self.Castling[0][0]==1 or self.Castling[0][1]==1)):
            castlingMoves,castlingNNinp=self.getCastlingMoves(len(outOfCheckMoves),debug,noOutputMode)
            if(debug): print("DEBUG (BoardPositions (findAllowedMoves)): castlingMoves=" ,castlingMoves)
            for i,move in enumerate(castlingMoves):
                outOfCheckMoves.append(move)
                outOfCheckNNinp.append(castlingNNinp[i])
        if noOutputMode==False: print("self.EnPassant=",self.EnPassant)
        if(self.EnPassant!=-1):
            enPassantMoves,enPassantNNinp=self.getEnPassantMoves(len(outOfCheckMoves),ownPositions[5],debug,noOutputMode)
            if(debug): print("DEBUG (BoardPositions (findAllowedMoves)): enPassantMoves=",enPassantMoves)
            for move in enPassantMoves:
                outOfCheckMoves.append(move)
                outOfCheckNNinp.append(enPassantNNinp[i])
        if len(outOfCheckMoves)>0:
            moveProbabilites=functions.moveProbs(self,Sizes,Weights,Biases,outOfCheckNNinp, debug)
        else:
            moveProbabilities=np.ones((1,1),dtype=np.float64)
        return outOfCheckMoves,moveProbabilites

    def nextPly(self):
        # Reverse the board
        if(  self.Reversed==True ): self.Reversed=False
        elif(self.Reversed==False): self.Reversed=True
        self.ChessBoard[:]=self.ChessBoard[::-1]
        self.PieceBoards=np.ascontiguousarray(self.PieceBoards[:,:,::-1])
        # Reverse the castling properties
        self.Castling[:]=self.Castling[::-1]
        self.KingPositions[:]=self.KingPositions[::-1]
        helperPlayer=self.CurrentOpponent
        self.CurrentOpponent=self.CurrentPlayer
        self.CurrentPlayer=helperPlayer
        # Increas ply number
        self.PlyNumber+=1
                            
    def playMove(self,move, writeMoveInfoList=False, debug=False, noOutputMode=False):
        player,opponent=self.CurrentPlayer,self.CurrentOpponent
        playerNumerator=functions.getPlayerNumerator(player)
        oppNumerator=functions.getPlayerNumerator(opponent)
        piece,pieceNew=move[0], move[1]
        oldPos=[move[2],move[3]]
        newPos=[move[4],move[5]] 
        capturedPiece=move[6] #TK
        isCastling =[move[7],move[8]] #TK
        isEnPassant=move[9]
        returnStringList=[]
        if(writeMoveInfoList):
            oldPosReadable=functions.getReadablePosition(player,oldPos[0],oldPos[1],debug)
            newPosReadable=functions.getReadablePosition(player,newPos[0],newPos[1],debug)
            returnStringList.append("Player "+str(player)+": "+functions.getPieceName(piece)+" from "+oldPosReadable+" to "+newPosReadable)
        if  (capturedPiece!=0 and isEnPassant!=0):
            if(noOutputMode==False and debug): print("INFO: Player",player,"just captured via en-passant at",oldPos)
            oppPawn=[newPos[0],newPos[1]-1]
            self.removePiece(opponent,oppNumerator, 6, oppPawn, False)
            capturedPiece=6
            if(writeMoveInfoList):
                returnStringList.append("Player "+str(player)+" captured "+functions.getPieceName(capturedPiece)+" via en-passant")
        elif(capturedPiece!=0 and isEnPassant==0):
            if(writeMoveInfoList):
                returnStringList.append("Player "+str(player)+" captured "+functions.getPieceName(capturedPiece)+" at "+newPosReadable)
            for pieceOpp in self.Pieces:
                self.removePiece(opponent,oppNumerator, pieceOpp, newPos, False)
        elif(capturedPiece==0 and isEnPassant!=0):
            print("ERROR: En-passant is always capture move!")
        self.removePiece(player,playerNumerator, piece, oldPos, False)
        self.setPiece(player,playerNumerator, pieceNew, newPos, False)
        if(pieceNew!=piece):
            if(piece==6):
                if(writeMoveInfoList): returnStringList.append("Player "+str(player)+"'s promoted his pawn to a "+functions.getPieceName(pieceNew)+" at "+newPosReadable)
            else:
                print("ERROR: Only pawns can be promoted.")
        if  (isCastling[0]==1): # castling long
            if(noOutputMode==False and debug): print("Player",player,"is CASTLING long!")
            self.removePiece(player,playerNumerator, 3, [0,0], False)
            self.setPiece(   player,playerNumerator, 3, [3,0], False)
            self.Castling[0]=[0,0]
            if(writeMoveInfoList): returnStringList.append("Player "+str(player)+" castled long.")
            if(isEnPassant!=0):print("ERROR: Impossible to be en-passant and castle move at the same time!")
        elif(isCastling[1]==1): # castling short
            if(noOutputMode==False and debug): print("Player",player,"is CASTLING short!")
            self.removePiece(player,playerNumerator, 3, [7,0], False)
            self.setPiece(   player,playerNumerator, 3, [5,0], False)
            self.Castling[0]=[0,0]
            if(writeMoveInfoList): returnStringList.append("Player "+str(player)+" castled short.")
            if(isEnPassant!=0):print("ERROR: Impossible to be en-passant and castle move at the same time!")
        if(self.Castling[0][0]==1 or self.Castling[0][1]==1):
            if(piece==1 and oldPos==[4,0]):
                if(noOutputMode==False and writeMoveInfoList==True):
                    print("INFO: Player",player,"is no longer allowed to castle.")
                self.Castling[0]=[0,0]
            elif(piece==3 and oldPos==[0,0]):
                if(self.Castling[0][0]==1):
                    if(noOutputMode==False and writeMoveInfoList==True):
                        print("INFO: Player",player,"is no longer allowed to castle long.")
                self.Castling[0][0]=0
            elif(piece==3 and oldPos==[7,0]):
                if(self.Castling[0][1]==1):
                    if(noOutputMode==False and writeMoveInfoList==True):
                        print("INFO: Player",player,"is no longer allowed to castle short.")
                self.Castling[0][1]=0
        if(piece==6 and oldPos[1]==1 and newPos[1]==3 and (self.ChessBoard[newPos[1]][newPos[0]-1]==opponent*6 or self.ChessBoard[newPos[1]][newPos[0]+1]==opponent*6)):
            self.EnPassant=newPos[0]
        else:
            self.EnPassant=-1
        return returnStringList,capturedPiece

    def reverseMove(self,move,capturedPiece,castlingBeforeMove,enPassantBeforeMove,isPlayerInCheckBeforeMove):
        player,opponent=self.CurrentPlayer,self.CurrentOpponent
        playerNumerator=functions.getPlayerNumerator(player)
        oppNumerator=functions.getPlayerNumerator(opponent)
        piece,pieceNew=move[0],move[1]
        oldPos=[move[2],move[3]]
        newPos=[move[4],move[5]] 
        capturedPiece=move[6]
        isCastling =[move[7],move[8]]
        isEnPassant=move[9]
        self.removePiece(player,playerNumerator, pieceNew, newPos, False)
        self.setPiece(player,playerNumerator, piece, oldPos, False)
        if  (capturedPiece!=0 and isEnPassant!=0):
            oppPawn=[newPos[0],newPos[1]-1]
            self.setPiece(opponent,oppNumerator, 6, oppPawn, False)
        elif(capturedPiece!=0 and isEnPassant==0):
            self.setPiece(opponent,oppNumerator, capturedPiece, newPos, False)
        if  (isCastling[0]==1): # castling long
            self.setPiece(   player,playerNumerator, 3, [0,0], False)
            self.removePiece(player,playerNumerator, 3, [3,0], False)
        elif(isCastling[1]==1): # castling short
            self.setPiece(   player,playerNumerator, 3, [7,0], False)
            self.removePiece(player,playerNumerator, 3, [5,0], False)
        self.Castling=castlingBeforeMove
        self.EnPassant=enPassantBeforeMove
        self.IsPlayerInCheck=isPlayerInCheckBeforeMove

    def setIsPlayerInCheck(self):
        playerIsInCheck=False
        #kingPosition=self.KingPositions[self.CurrentPlayer] #TK changed
        kingPosition=self.KingPositions[0]
        ownSign,oppSign=self.CurrentPlayer,self.CurrentOpponent
        pos=[0,0]
        BasicMoves=[[[1,1],[-1,1],[1,-1],[-1,-1]], #bishopMoves
                    [[1,0],[-1,0],[0, 1], [0,-1]], #rookMoves
                    [[1,2],[-1,2],[1,-2],[-1,-2],[2,1],[-2,1],[2,-1],[-2,-1]], #knightMoves
                    [[1,1],[-1,1]]] #pawnMoves
        basicMoves=nb.typed.List(BasicMoves)
        for num,moves in enumerate(basicMoves):
            for move in moves:
                i=1
                while i<=8:
                    pos[0]=kingPosition[0]+move[0]*i
                    pos[1]=kingPosition[1]+move[1]*i
                    if(pos[0]<0 or pos[0]>7 or pos[1]<0 or pos[1]>7 or (ownSign*self.ChessBoard[pos[1]][pos[0]]>0)):
                        break
                    elif(oppSign*self.ChessBoard[pos[1]][pos[0]]>0):
                        oppK,oppQ,oppR,oppB,oppN,oppP=oppSign*1,oppSign*2,oppSign*3,oppSign*4,oppSign*5,oppSign*6
                        if((num==0 and (self.ChessBoard[pos[1]][pos[0]]==oppQ or self.ChessBoard[pos[1]][pos[0]]==oppB or (i==1 and self.ChessBoard[pos[1]][pos[0]]==oppK))) or
                           (num==1 and (self.ChessBoard[pos[1]][pos[0]]==oppQ or self.ChessBoard[pos[1]][pos[0]]==oppR or (i==1 and self.ChessBoard[pos[1]][pos[0]]==oppK))) or
                           (num==2 and  self.ChessBoard[pos[1]][pos[0]]==oppN) or
                           (num==3 and  self.ChessBoard[pos[1]][pos[0]]==oppP)):
                            playerIsInCheck=True
                        elif(num==0 or num==1):
                            break
                    if(num==2 or num==3):
                        break
                    i+=1
        self.IsPlayerInCheck=playerIsInCheck
                    
    def removePiece(self, player,playerNumerator, piece, oldPos, debug=False):
        pieceNumerator=piece-1
        self.PieceBoards[playerNumerator][pieceNumerator][oldPos[1]][oldPos[0]]=0
        self.ChessBoard[oldPos[1]][oldPos[0]]=0

    def setPiece(self, player,playerNumerator, piece, newPos, debug=False):
        pieceNumerator=piece-1
        self.PieceBoards[playerNumerator][pieceNumerator][newPos[1]][newPos[0]]=1
        self.ChessBoard[newPos[1]][newPos[0]]=player*piece
        if(piece==1):
            self.KingPositions[0]=newPos

    def setWinner(self,winner):
        self.Winner=winner

    def getWinner(self):
        return self.Winner

    def getInput(self):
        pB=self.PieceBoards
        #print("pB=",pB)
        if(self.CurrentPlayer==+1): #white
            #flattenedInput=np.vstack((pB[0][0].ravel(),pB[0][1].ravel(),pB[0][2].ravel(),pB[0][3].ravel(),pB[0][4].ravel(),pB[0][5].ravel(),\
            #                          pB[1][0].ravel(),pB[1][1].ravel(),pB[1][2].ravel(),pB[1][3].ravel(),pB[1][4].ravel(),pB[1][5].ravel())).ravel()
            flattenedInput=pB.reshape(768,)
        elif(self.CurrentPlayer==-1): #Black
            #flattenedInput=np.vstack((pB[1][0].ravel(),pB[1][1].ravel(),pB[1][2].ravel(),pB[1][3].ravel(),pB[1][4].ravel(),pB[1][5].ravel(),\
            #                          pB[0][0].ravel(),pB[0][1].ravel(),pB[0][2].ravel(),pB[0][3].ravel(),pB[0][4].ravel(),pB[0][5].ravel())).ravel()
            flattenedInput=np.ascontiguousarray(pB[::-1]).reshape(768,)
        else:
            flattenedInput=np.vstack((pB[0][0].ravel(),pB[0][1].ravel(),pB[0][2].ravel(),pB[0][3].ravel(),pB[0][4].ravel(),pB[0][5].ravel(),\
                                    pB[1][0].ravel(),pB[1][1].ravel(),pB[1][2].ravel(),pB[1][3].ravel(),pB[1][4].ravel(),pB[1][5].ravel())).ravel()
            print("ERROR (BoardPositions (getInput): Wrong player input=",self.CurrentPlayer)
        castlingArray=np.zeros(4,dtype=np.int64)
        enpassantArray=np.zeros(8,dtype=np.int64)
        if(self.Castling[0][0]==1): castlingArray[0]=1
        if(self.Castling[0][1]==1): castlingArray[1]=1
        if(self.Castling[1][0]==1): castlingArray[2]=1
        if(self.Castling[1][1]==1): castlingArray[3]=1
        if(self.EnPassant!=-1): enpassantArray[self.EnPassant]=1
        flattenedInput=np.concatenate((flattenedInput,enpassantArray,castlingArray))
        #print("flattenedInput=",flattenedInput)
        return flattenedInput

    def reconstructGameState(self,idx,flattenedInput,plyNumber):
        if plyNumber%2==1:
            self.CurrentPlayer,self.CurrentOpponent=+1,-1
            self.Reversed = False
        elif plyNumber%2==0:
            self.CurrentPlayer,self.CurrentOpponent=-1,+1
            self.Reversed = True
        self.PlyNumber=plyNumber
        if self.CurrentPlayer==+1:
            self.PieceBoards=flattenedInput[0:768].reshape((2,6,8,8))
        else:
            self.PieceBoards=np.ascontiguousarray(flattenedInput[0:768].reshape((2,6,8,8))[::-1])
        #self.PieceBoards=flattenedInput[0:768].reshape((2,6,8,8))
        #if self.CurrentPlayer==-1:
        #    for playerNum,playerPieceBoards in enumerate(self.PieceBoards):
        #        for pieceNumerator,piece in enumerate(playerPieceBoards):
        #            self.PieceBoards[playerNum][pieceNumerator][:]=piece[::-1]
        self.setChessBoard(idx)
        enPassantArray=flattenedInput[768:776]
        self.EnPassant = -1
        for i,elem in enumerate(enPassantArray):
            if elem==1:
                self.EnPassant=i
        castlingArray =flattenedInput[776:780]
        self.Castling[0][0]=castlingArray[0]
        self.Castling[0][1]=castlingArray[1]
        self.Castling[1][0]=castlingArray[2]
        self.Castling[1][1]=castlingArray[3]
        self.Finished=False
        if idx==0:
            print("flattenedInput=",flattenedInput)
            print("castlingArray=",castlingArray)
            print("self.EnPassant=",self.EnPassant)
            print("self.CurrentPlayer,self.CurrentOpponent,self.Finished=",self.CurrentPlayer,self.CurrentOpponent,self.Finished)

    def getInputID(self,debug):
        if(debug): print("ChessBoard=",self.ChessBoard)
        playerSign=self.CurrentPlayer
        oppSign   =self.CurrentOpponent
        InputIDString=""
        for rowNum,row in enumerate(self.ChessBoard):
            countEmpty=0
            for colNum,col in enumerate(row):
                if countEmpty==0:
                    num=""
                else:
                    num=str(countEmpty)
                if(playerSign*col>0):
                    InputIDString+=num+functions.getPieceName(col)[0]
                    countEmpty=0
                elif(oppSign*col>0):
                    InputIDString+=num+functions.getPieceName(col)[0].upper()
                    countEmpty=0
                else:
                    countEmpty+=1
            if countEmpty>0 and countEmpty<8:
                InputIDString+=str(countEmpty)
            if(rowNum<7):
                InputIDString+="/"
        if(self.Castling[0][0]==1 or self.Castling[0][1]==1 or self.Castling[1][0]==1 or self.Castling[1][1]==1):
            InputIDString+="_"
            if(self.Castling[0][0]==1): InputIDString+="l"
            if(self.Castling[0][1]==1): InputIDString+="s"
            if(self.Castling[1][0]==1): InputIDString+="L"
            if(self.Castling[1][1]==1): InputIDString+="S"
        if(self.EnPassant!=-1):
            xpos_plus=self.EnPassant+1
            xpos_minus=self.EnPassant-1
            ownSign=self.CurrentPlayer
            if((xpos_plus<=7 and ownSign*self.ChessBoard[4][xpos_plus]==6) or (xpos_minus>=0 and ownSign*self.ChessBoard[4][xpos_minus]==6)):
                InputIDString+="_"+str(self.EnPassant)
        if(debug): print("InputIDString =",InputIDString)
        return InputIDString
