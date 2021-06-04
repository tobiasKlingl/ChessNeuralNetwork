import numpy as np
import ChessBoard as cb
import moves
import functions
#from colorama import init, Fore, Back, Style
import time

class BoardPositions(object):
    def __init__(self, num_rows=8, num_columns=8, num_pieces=6):
        self.PlayerColors = ["White","Black"]
        self.Pieces = ["pawn","knight","bishop","rook","queen","king"]
        self.PieceBoards = []
        self.NumRows = num_rows
        self.NumColumns = num_columns
        self.NumChessField = self.NumRows*self.NumColumns
        self.NumPieces = len(self.Pieces)
        self.Reversed = False
        self.KingPositions = [[4,0],[4,0]]   # from current players view (black's rows are inverted to have a common description)
        self.Winner = -1
        
    def definePieceBoards(self, debug=False, noOutputMode=False):
        if(debug): print("DEBUG (BoardPositions (definePieceBoards)): Define PieceBoards.")
        for i,playerColor in enumerate(self.PlayerColors):
            playerBoards = []
            for j,piece in enumerate(self.Pieces):
                if(debug): print("DEBUG (BoardPositions (definePieceBoards)): Adding pieceBoard for",playerColor,piece)
                pieceBoard = cb.ChessBoard(playerColor, piece, self.NumRows, self.NumColumns, self.NumPieces).pieceBoardMatrix
                playerBoards.append(pieceBoard)
            self.PieceBoards.append(playerBoards)
        if(debug): print("DEBUG (BoardPositions (definePieceBoards)): All pieceBoards defined and added to self.PieceBoards:",self.PieceBoards)

    def pieceInitializationInformation(self,playerColor, pieceName, debug=False):
        pieceArray = [0,0,0,0,0,0,0,0]
        rowNum = -1
        if playerColor == "White":
            rowNumPawns = 1 #white pawns are in row 1
            rowNumPieces= 0 #white pieces are in row 0
        elif playerColor == "Black":
            rowNumPawns = 6 #black pawns are in row 6
            rowNumPieces= 7 #black pieces are in row 7
        else:
            print("ERROR (BoardPositions (pieceInitializationInformation)): Player playerColor unknown.")
        if pieceName == "pawn":
            pieceArray = [1,1,1,1,1,1,1,1]
            rowNum = rowNumPawns
        elif pieceName == "knight":
            pieceArray = [0,1,0,0,0,0,1,0]
            rowNum = rowNumPieces
        elif pieceName == "bishop":
            pieceArray = [0,0,1,0,0,1,0,0]
            rowNum = rowNumPieces
        elif pieceName == "rook":
            pieceArray = [1,0,0,0,0,0,0,1]
            rowNum = rowNumPieces
        elif pieceName == "queen":
            pieceArray = [0,0,0,1,0,0,0,0]
            rowNum = rowNumPieces
        elif pieceName == "king":
            pieceArray = [0,0,0,0,1,0,0,0]
            rowNum = rowNumPieces
        else:
            print("ERROR (BoardPositions (pieceInitializationInformation)): pieceName unknown!")
        if(debug): print("DEBUG (BoardPositions (pieceInitializationInformation)): chess board row rumber:", pieceArray,"/",rowNum)
        return pieceArray, rowNum

    def initializeBoard(self, colored, debug=False, noOutputMode=False):
        if(noOutputMode==False):
            print("\n ###################################")
            print(" #### Initializing piece boards! ####")
            print(" ###################################")
            self.printChessBoard(0,colored,[],False)
        if(debug):
            print("DEBUG (BoardPositions (initializeBoard)): Number of rows/Columns = ",self.NumRows,", ",self.NumColumns)
            print("Number of all chess boards positions:",len(self.PlayerColors)*self.NumChessField*self.NumPieces)
        for player,playerColor in enumerate(self.PlayerColors):
            if (debug): print("DEBUG (BoardPositions (initializeBoard)): Initializing",playerColor,"pieces!")
            for piece_number,piece_name in enumerate(self.Pieces):
                if(debug): print("DEBUG (BoardPositions (initializeBoard)) Getting",playerColor,piece_name,"Initializing information.")
                pieceInitializationArray, rowNum = self.pieceInitializationInformation(playerColor, piece_name, debug)
                self.PieceBoards[player][piece_number][rowNum] = pieceInitializationArray
            if(noOutputMode==False):
                print("DEBUG (BoardPositions (initializeBoard)):",playerColor,"pieces initialized!")
                self.printChessBoard(0,colored,[],False)
        if(noOutputMode==False):
            print("\n ###################################")
            print(" #### All pieces initialized! ######")
            print(" ################################### \n")

    def getPositions(self,pieceBoards, debug=False): #get the positions of pieces [col,row]
        piecePositions=[]
        for pieceNum,piece in enumerate(self.Pieces):
            if(debug): print("DEBUG (BoardPositions (getPositions)): Current piece:",piece,"(",pieceNum,")")
            currentPositions=[]
            for rowNum,row in enumerate(pieceBoards[pieceNum]):
                if(debug): print("DEBUG (BoardPositions (getPositions)): row=",row,"(",rowNum,")")
                for colNum,col in enumerate(pieceBoards[pieceNum][rowNum]):
                    if(debug): print("DEBUG (BoardPositions (getPositions)): col=",col,"(",colNum,")")
                    if col==1:
                        currentPositions.append([colNum,rowNum])
            piecePositions.append(currentPositions)
        return piecePositions

    def getPieceNumber(self,pieceName):
        pieceNumber=0
        if pieceName=="pawn":
            pieceNumber=0
        elif pieceName=="knight":
            pieceNumber=1
        elif pieceName=="bishop":
            pieceNumber=2
        elif pieceName=="rook":
            pieceNumber=3
        elif pieceName=="queen":
            pieceNumber=4
        elif pieceName=="king":
            pieceNumber=5
        return pieceNumber

    def getPiecePositions(self,player,debug=False):
        if player==0:
            ownPieceBoards=self.PieceBoards[0]
            oppPieceBoards=self.PieceBoards[1]
        elif player==1:
            ownPieceBoards=self.PieceBoards[1]
            oppPieceBoards=self.PieceBoards[0]
        else:
            print("(ERROR) Invalid player.")
        ownPiecePositions=self.getPositions(ownPieceBoards)
        oppPiecePositions=self.getPositions(oppPieceBoards)
        if(debug):
            print("DEBUG (BoardPositions (getPiecePositions)): Own Pieceboards of player:",player,":",ownPieceBoards)
            print("DEBUG (BoardPositions (getPiecePositions)): Opp Pieceboards of player:",player,":",oppPieceBoards)
            print("DEBUG (BoardPositions (getPiecePositions)): Own piece positions of player:",player,":",ownPieceBoards)
            print("DEBUG (BoardPositions (getPiecePositions)): Opp piese positions of player:",player,":",oppPieceBoards)
        return ownPiecePositions,oppPiecePositions

    def getNormalMoves(self,player,chessBoard,ownPositions,debug=False):
        normalMoves=[]
        onlyCaptureMoves=False
        for pieceNum,piece in enumerate(self.Pieces):
            if(debug): print("DEBUG (BoardPositions (getNormalMoves)): Current piece=",piece,"; ownPositions=",ownPositions[pieceNum])
            allowed=[]
            for piecePosition_i in ownPositions[pieceNum]:
                movesForPiece=moves.getAllowedMovesForPiece(player,piece,piecePosition_i,chessBoard,onlyCaptureMoves,debug)
                if(debug): print("DEBUG (BoardPositions (getNormalMoves)): Moves for",piece,"=",movesForPiece)
                if(movesForPiece): normalMoves.append(movesForPiece)
        normalMovesList=[item for sublist in normalMoves for item in sublist]
        return normalMovesList

    def getCastlingMoves(self,player,castling,chessBoard,debug):
        castlingMoves=[]
        kingPosition=self.KingPositions[player]
        if(kingPosition==[4,0]):
            if(castling[0]==True and chessBoard[0][1]=="  " and chessBoard[0][2]=="  " and chessBoard[0][3]=="  "): #castling long
                if(debug): print("DEBUG (BoardPositions (getCastlingMoves)): Adding castling LONG to player",player,"'s castlingMoves.")
                castlingMoves.append([["king","king",kingPosition,[2,0],False]])
            if(castling[1]==True and chessBoard[0][6]=="  " and chessBoard[0][5]=="  "): #castling short
                if(debug): print("DEBUG (BoardPositions (getCastlingMoves)): Adding castling SHORT to player",player,"'s castlingMoves.")
                castlingMoves.append([["king","king",kingPosition,[6,0],False]])
        castlingMovesList=[item for sublist in castlingMoves for item in sublist]
        return castlingMovesList

    def getEnPassantMoves(self,player,pawnPositions,oppPawn,debug=False):
        enPassantMoves=[]
        for pawn in pawnPositions:
            if(pawn[1]==4 and oppPawn[1]==4 and (pawn[0]==oppPawn[0]-1 or pawn[0]==oppPawn[0]+1)):
                print("INFO: En-passant move available for player",player,"'s pawn at",pawn)
                position_after=[oppPawn[0],oppPawn[1]+1]
                enPassantMoves.append([["pawn","pawn",pawn,position_after,True]])
        enPassantMovesList=[item for sublist in enPassantMoves for item in sublist]
        return enPassantMovesList
    
    def findAllowedMoves(self, player, castling, enPassant, debug=False):
        ownPositions,oppPositions=self.getPiecePositions(player,debug)
        chessBoard=self.getChessBoard(False)
        normalMoves=self.getNormalMoves(player,chessBoard,ownPositions,debug)
        if(castling[0]==True or castling[1]==True):
            castlingMoves=self.getCastlingMoves(player,castling,chessBoard,debug)
        else:
            castlingMoves=[]
        if(enPassant[0]==True):
            oppPawnPositionOppPerspective=enPassant[1]
            oppPawn=[oppPawnPositionOppPerspective[0],7-oppPawnPositionOppPerspective[1]]
            enPassantMoves=self.getEnPassantMoves(player,ownPositions[0],oppPawn,debug)
        else:
            enPassantMoves=[]
        if(debug):
            print("DEBUG (BoardPositions (findAllowedMoves)): normalMoves=",normalMoves)
            print("DEBUG (BoardPositions (findAllowedMoves)): castlingMoves=",castlingMoves)
        return normalMoves,castlingMoves,enPassantMoves
               
    def reverseBoard(self):
        if(  self.Reversed==True ): self.Reversed=False
        elif(self.Reversed==False): self.Reversed=True
        for playerNum,player in enumerate(self.PieceBoards):
            for pieceNum,piece in enumerate(player):
                self.PieceBoards[playerNum][pieceNum]=np.flip(self.PieceBoards[playerNum][pieceNum],0)

    def getKingPosition(self,playerNum,debug=False):
        position=self.KingPositions[playerNum]
        if(debug): print("DEBUG (BoardPositions (getKingPosition)): Player",playerNum,"'s kingPosition=",position)
        return position
    
    def printChessBoard(self, player, colored=False, moveInfoList=[], debug=False):
        if(debug): print("moveInfoList=",moveInfoList)
        if(len(moveInfoList)>2):
            print("ERROR: len(moveInfoList)=",len(moveInfoList),"shouldn't be larger than 2!")
        chessBoard=self.getChessBoard(False)
        if(player==0):
            chessBoard=np.flip(chessBoard,0)
        print("###### Board- ######")
        print("## A B C D E F G H #")
        bkgColor="49"
        for colNum,column in enumerate(chessBoard):
            print(8-colNum,end=' ')
            printColor=""
            for i in range(8):
                if colored:
                    if  ((8-colNum)%2==0 and i%2==0) or ((8-colNum)%2==1 and i%2==1): bkgColor="44"
                    elif((8-colNum)%2==0 and i%2==1) or ((8-colNum)%2==1 and i%2==0): bkgColor="46"
                    if   "B" in column[i]: printColor="\033[0;30;"+bkgColor+"m"
                    elif "W" in column[i]: printColor="\033[0;37;"+bkgColor+"m"
                    else: printColor="\033[1;37;"+bkgColor+"m"
                pieceUnicode=functions.getPieceUnicode(column[i],colored)
                print(printColor+pieceUnicode+" ",end='')
            if(colored):
                if(colNum==2 and len(moveInfoList)>0):
                    print("\033[1;37;49m",8-colNum,"  ",moveInfoList[0])
                elif(colNum==4 and len(moveInfoList)>1):
                    print("\033[1;37;49m",8-colNum,"  ",moveInfoList[1])
                else:
                    print("\033[1;37;49m",8-colNum)
            else:
                if(colNum==2 and len(moveInfoList)>0):
                    print(8-colNum,"  ",moveInfoList[0])
                elif(colNum==4 and len(moveInfoList)>1):
                    print(8-colNum,"  ",moveInfoList[1])
                else:
                    print(8-colNum)
        print("# A B C D E F G H ##")
        print("###### -Board ######")
            
    def getChessBoard(self, debug=False):
        chessBoard = np.array([]), np.array([])
        for i in range (self.NumRows*self.NumColumns):
            chessBoard = np.append(chessBoard,"  ")
        chessBoard.shape = (self.NumRows, self.NumColumns)
        for player,playerColor in enumerate(self.PlayerColors):
            playerabb = playerColor[:1]
            for pieceNumber, pieceName in enumerate(self.Pieces):
                if pieceName != "knight":
                    pieceAbb = pieceName[:1]
                else:
                    pieceAbb = "n"
                for row in range(self.NumRows):
                    for col in range(self.NumColumns):
                        if(self.PieceBoards[player][pieceNumber][row][col] == 1):
                            chessBoard[row][col] = playerabb+pieceAbb
        return chessBoard

    def playMove(self,player, move, castle, isEnPassant, writeMoveInfoList=False, debug=False):
        piece=move[0]
        pieceNew=move[1]
        oldPos=move[2]
        newPos=move[3]
        capturedMove=move[4]
        returnStringList=[]
        if(writeMoveInfoList):
            oldPosReadable=functions.getReadablePosition(player,move[2],debug)
            newPosReadable=functions.getReadablePosition(player,move[3],debug)
            returnStringList.append("Player "+str(player)+": "+piece+" from "+oldPosReadable+" to "+newPosReadable)
        if  (capturedMove==True and isEnPassant==True):
            print("isEnPassant=",isEnPassant)
            print("newPos=",newPos)
            opponent=functions.getOpponent(player)
            oppPawn=[newPos[0],newPos[1]-1]
            self.removePiece(opponent, "pawn", oppPawn, False)
            if(writeMoveInfoList): returnStringList.append("Player "+str(player)+" captured player "+str(opponent)+"'s piece via en-passant")
        elif(capturedMove==True and isEnPassant==False):
            opponent=functions.getOpponent(player)
            if(writeMoveInfoList): returnStringList.append("Player "+str(player)+" captured player "+str(opponent)+"'s piece at "+newPosReadable)
            for pieceOpp in self.Pieces: self.removePiece(opponent, pieceOpp, newPos, False)
        elif(capturedMove==False and isEnPassant==True):
            print("ERROR: En-passant is always capture move!")
        self.removePiece(player, piece, oldPos, False)
        self.setPiece(player, pieceNew, newPos, False)
        if(pieceNew!=piece):
            if(piece=="pawn"):
                if(writeMoveInfoList): returnStringList.append("Player "+str(player)+"'s promoted his pawn to a "+pieceNew+" at "+newPosReadable)
            else:
                print("ERROR: Only pawns can be promoted.")
        if  (castle[0]==True): # castling long
            self.removePiece(player, "rook", [0,0], False)
            self.setPiece(   player, "rook", [3,0], False)
            if(writeMoveInfoList): returnStringList.append("Player "+str(player)+" castled long.")
            if(isEnPassant==True):print("ERROR: Impossible to be en-passant and castle move at the same time!")
        elif(castle[1]==True): # castling short
            self.removePiece(player, "rook", [7,0], False)
            self.setPiece(   player, "rook", [5,0], False)
            if(writeMoveInfoList): returnStringList.append("Player "+str(player)+" castled short.")
            if(isEnPassant==True):print("ERROR: Impossible to be en-passant and castle move at the same time!")
        return returnStringList
            
    def isPlayerInCheck(self,player, debug=False):
        playerIsInCheck=False
        kingPosition=self.KingPositions[player]
        if(debug): print("DEBUG (BoardPositions (isPlayerInCheck)): King position of player",player,"is",kingPosition)
        chessBoard=self.getChessBoard()
        if(debug):
            print("(BoardPositions (isPlayerInCheck)):")
            self.printChessBoard(player)
        if(debug): print("(BoardPositions (isPlayerInCheck)): chessBoard[",kingPosition[0],"][",kingPosition[1],"]=\n",chessBoard[kingPosition[1]][kingPosition[0]])
        ownColor=self.PlayerColors[player][0]
        oppColor=self.PlayerColors[functions.getOpponent(player)][0]
        pos=[0,0]
        bishopMoves=[[1,1],[-1,1],[1,-1],[-1,-1]]
        rookMoves  =[[1,0],[-1,0], [0,1], [0,-1]]
        knightMoves=[[1,2],[-1,2],[1,-2],[-1,-2],[2,1],[-2,1],[2,-1],[-2,-1]]
        pawnMoves=[[1,1],[-1,1]]
        basicMoves=[bishopMoves,rookMoves,knightMoves,pawnMoves]
        for num,moves in enumerate(basicMoves):
            if  (num==0 and debug): print("DEBUG (BoardPositions (isPlayerInCheck)): Check bishop moves!")
            elif(num==1 and debug): print("DEBUG (BoardPositions (isPlayerInCheck)): Check rook moves!")
            elif(num==2 and debug): print("DEBUG (BoardPositions (isPlayerInCheck)): Check knight moves!")
            elif(num==3 and debug): print("DEBUG (BoardPositions (isPlayerInCheck)): Check pawn moves!")
            for move in moves:
                if(debug): print("DEBUG (BoardPositions (isPlayerInCheck)): move=",move)
                for i in range(1,8):
                    pos[0]=kingPosition[0]+move[0]*i
                    pos[1]=kingPosition[1]+move[1]*i
                    if(pos[0]<0 or pos[0]>7 or pos[1]<0 or pos[1]>7):
                        if(debug):
                            print("DEBUG (BoardPositions (isPlayerInCheck)): pos[0]=kingPosition[0]+delt[0]=",kingPosition[0],"+",move[0]*i,"=",pos[0])
                            print("DEBUG (BoardPositions (isPlayerInCheck)): pos[1]=kingPosition[1]+delt[1]=",kingPosition[1],"+",move[1]*i,"=",pos[1])
                            print("DEBUG (BoardPositions (isPlayerInCheck)): Position is out of bounds")
                        break
                    elif(ownColor in chessBoard[pos[1]][pos[0]]):
                        if(debug):
                            print("DEBUG (BoardPositions (isPlayerInCheck)): pos[0]=kingPosition[0]+delt[0]=",kingPosition[0],"+",move[0]*i,"=",pos[0])
                            print("DEBUG (BoardPositions (isPlayerInCheck)): pos[1]=kingPosition[1]+delt[1]=",kingPosition[1],"+",move[1]*i,"=",pos[1])
                            print("DEBUG (BoardPositions (isPlayerInCheck)): Player",player,"'s king has his own piece at [",pos[0],",",pos[1],"]")
                        break
                    elif(oppColor in chessBoard[pos[1]][pos[0]]):
                        oppQ=oppColor+"q"
                        oppB=oppColor+"b"
                        oppR=oppColor+"r"
                        oppN=oppColor+"n"
                        oppP=oppColor+"p"
                        oppK=oppColor+"k"
                        if  (num==0 and (chessBoard[pos[1]][pos[0]]==oppQ or chessBoard[pos[1]][pos[0]]==oppB or (i==1 and chessBoard[pos[1]][pos[0]]==oppK))): return True
                        elif(num==1 and (chessBoard[pos[1]][pos[0]]==oppQ or chessBoard[pos[1]][pos[0]]==oppR or (i==1 and chessBoard[pos[1]][pos[0]]==oppK))): return True
                        elif(num==2 and chessBoard[pos[1]][pos[0]]==oppN): return True
                        elif(num==3 and chessBoard[pos[1]][pos[0]]==oppP): return True
                        elif((num==0 or num==1) and oppColor in chessBoard[pos[1]][pos[0]]):
                            if(debug):
                                print("DEBUG (BoardPositions (isPlayerInCheck)): pos[0]=kingPosition[0]+delt[0]=",kingPosition[0],"+",move[0]*i,"=",pos[0])
                                print("DEBUG (BoardPositions (isPlayerInCheck)): pos[1]=kingPosition[1]+delt[1]=",kingPosition[1],"+",move[1]*i,"=",pos[1])
                                print("DEBUG (BoardPositions (isPlayerInCheck)): Player",player,"has a black non-attacking piece (knight or pawn) at [",pos[0],",",pos[1],"]")
                            break
                    if(num==2 or num==3):
                        break
                
        return False

    def removePiece(self, player, piece, oldPos, debug=False):
        pieceNumber=self.getPieceNumber(piece)
        self.PieceBoards[player][pieceNumber][oldPos[1]][oldPos[0]]=0
        
    def setPiece(self, player, piece, newPos, debug=False):
        pieceNumber=self.getPieceNumber(piece)
        self.PieceBoards[player][pieceNumber][newPos[1]][newPos[0]]=1
        if(piece=="king"):
            self.KingPositions[player]=newPos

    def setWinner(self,winner):
        self.Winner=winner

    def getWinner(self):
        return self.Winner

    def getInput(self,player,enPassant,castling,debug,noOutputMode):
        pB=self.PieceBoards
        if(player==0):
            flattenedInput=np.vstack((pB[0][0].ravel(),pB[0][1].ravel(),pB[0][2].ravel(),pB[0][3].ravel(),pB[0][4].ravel(),pB[0][5].ravel(),\
                                      pB[1][0].ravel(),pB[1][1].ravel(),pB[1][2].ravel(),pB[1][3].ravel(),pB[1][4].ravel(),pB[1][5].ravel())).ravel()
        elif(player==1):
            flattenedInput=np.vstack((pB[1][0].ravel(),pB[1][1].ravel(),pB[1][2].ravel(),pB[1][3].ravel(),pB[1][4].ravel(),pB[1][5].ravel(),\
                                      pB[0][0].ravel(),pB[0][1].ravel(),pB[0][2].ravel(),pB[0][3].ravel(),pB[0][4].ravel(),pB[0][5].ravel())).ravel()
        else:
            print("ERROR (BoardPositions (getInput): Wrong player input=",player)
        #t=time.time()
        castlingArray=np.zeros(4)
        enpassantArray=np.zeros(8)
        #print("castling=",castling)
        if(castling[player][0]==True):
            castlingArray[0]=1
        if(castling[player][1]==True):
            castlingArray[1]=1
        opponent=functions.getOpponent(player)
        if(castling[opponent][0]==True):
            castlingArray[2]=1
        if(castling[opponent][1]==True):
            castlingArray[3]=1
        #print("enPassant=",enPassant)
        if(enPassant[0]==True):
            enpassantArray[enPassant[1][0]]=1
        flattenedInput=np.concatenate((flattenedInput,enpassantArray,castlingArray))
        #print("time for n.pzeros(0):",time.time()-t)
        return flattenedInput

    def getInputID(self,player,enPassant,castling,debug,noOutputMode):
        chessBoard=self.getChessBoard(debug)
        if(debug): print("ChessBoard=",chessBoard)
        opponent=functions.getOpponent(player)
        
        InputIDString=""
        for rowNum,row in enumerate(chessBoard):
            for colNum,col in enumerate(row):
                stringToAppend=""
                if(player==0 and "W" in col):
                    stringToAppend=str(colNum)+col[-1:]
                elif(player==0 and "B" in col):
                    stringToAppend=str(colNum)+col[-1:].upper()
                elif(player==1 and "B" in col):
                    stringToAppend=str(colNum)+col[-1:]
                elif(player==1 and "W" in col):
                    stringToAppend=str(colNum)+col[-1:].upper()
                InputIDString+=stringToAppend
            if(rowNum<7):
                InputIDString+="/"
        if(castling[player][0]==True or castling[player][1]==True or castling[opponent][0]==True or castling[opponent][1]==True):
            InputIDString+="_"
            if(castling[player][0]==True): InputIDString+="l"
            if(castling[player][1]==True): InputIDString+="s"
            if(castling[opponent][0]==True): InputIDString+="L"
            if(castling[opponent][1]==True): InputIDString+="S"
        if(enPassant[0]==True):
            InputIDString+="_"
            InputIDString+=str(enPassant[1][1])
        if(debug): print("InputIDString =",InputIDString)
        return InputIDString
