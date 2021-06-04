import numpy
import random
import copy
import time
#from colorama import init, Fore, Back, Style

def getPlayerColor(player):
    if  (player==0): color="W"
    elif(player==1): color="B"
    else:print("(ERROR (functions (getPlayerColor)): invalid argument for player=",player)
    return color

def getOpponent(player):
    if  (player==0):opponent=1
    elif(player==1):opponent=0
    else:print("(ERROR (function (getOpponent): invalid argument for player=",player)
    return opponent

def nextPly(plyNumber,player):
    plyNumber+=1
    player=getOpponent(player)
    return plyNumber,player

def getReadablePosition(player,move,debug=False):
    if  (move[0]==0): xpos="A"
    elif(move[0]==1): xpos="B"
    elif(move[0]==2): xpos="C"
    elif(move[0]==3): xpos="D"
    elif(move[0]==4): xpos="E"
    elif(move[0]==5): xpos="F"
    elif(move[0]==6): xpos="G"
    elif(move[0]==7): xpos="H"
    else:print("ERROR (functions (getReadablePosition)): move[0]=",move[0]," is unknown!")
    if  (player==0): ypos=str(move[1]+1)
    elif(player==1): ypos=str(7-move[1]+1)
    readablePosition=xpos+ypos
    if(debug): print("DEBUG (functions (getReadablePosition)): Converted position:",move,"into:",readablePosition)
    return readablePosition

def getPieceUnicode(pieceAbb,colored):
    pieceUnicode=" "
    if(colored==True):
        if  ("k" in pieceAbb): pieceUnicode='\u265A'
        elif("q" in pieceAbb): pieceUnicode='\u265B'
        elif("r" in pieceAbb): pieceUnicode='\u265C'
        elif("b" in pieceAbb): pieceUnicode='\u265D'
        elif("n" in pieceAbb): pieceUnicode='\u265E'
        elif("p" in pieceAbb): pieceUnicode='\u265F'
    else:
        if("W" in pieceAbb): x=6
        elif("B" in pieceAbb): x=0
        if  ("k" in pieceAbb): pieceUnicode="".join(chr(9812+0+x))
        elif("q" in pieceAbb): pieceUnicode="".join(chr(9812+1+x))
        elif("r" in pieceAbb): pieceUnicode="".join(chr(9812+2+x))
        elif("b" in pieceAbb): pieceUnicode="".join(chr(9812+3+x))
        elif("n" in pieceAbb): pieceUnicode="".join(chr(9812+4+x))
        elif("p" in pieceAbb): pieceUnicode="".join(chr(9812+5+x))
    return pieceUnicode

def createPieceListforPrintOut(readableMoveList,colored=False,debug=False):
    sortedList=[]
    pawnMoves  ="  PAWN: "
    knightMoves="  KNIGHT: "
    bishopMoves="  BISHOP: "
    rookMoves  ="  ROOK: "
    queenMoves ="  QUEEN: "
    kingMoves  ="  KING: "
    for ID,move in enumerate(readableMoveList):
        if colored:
            moveWithID="\033[1;34;49m"+move[1]+"->"+move[2]+"\033[1;37;49m("+str(ID)+") ; "
        else:
            moveWithID=move[1]+"->"+move[2]+"("+str(ID)+") ; "
        if  ("pawn"  in move): pawnMoves+=moveWithID
        elif("knight"in move): knightMoves+=moveWithID
        elif("bishop"in move): bishopMoves+=moveWithID
        elif("rook"  in move): rookMoves+=moveWithID
        elif("queen" in move): queenMoves+=moveWithID
        elif("king"  in move): kingMoves+=moveWithID
        else:
            print("ERROR (functions (createPieceListforPrintOut)): unknown pieceName")
    sortedList.append(pawnMoves)
    sortedList.append(knightMoves)
    sortedList.append(bishopMoves)
    sortedList.append(rookMoves)
    sortedList.append(queenMoves)
    sortedList.append(kingMoves)
    return sortedList

def printMoves(player,moveList,colored,noOutputMode=False,debug=False):
    readableMoveList=[]
    if(debug): print("moveList=",moveList)
    for move in moveList:
        piece=move[0]
        pieceNew=move[1]
        capturedMove=move[4]
        pos_before=getReadablePosition(player,move[2],debug)
        pos_after=getReadablePosition(player,move[3],debug)
        if(piece=="king" and pos_before=="E1" and (pos_after=="G1" or pos_after=="C1")):
            readableMove=[piece,pos_before,pos_after+"(White CASTLING)"]
        elif(piece=="king" and pos_before=="E8" and (pos_after=="G8" or pos_after=="C8")):
            readableMove=[piece,pos_before,pos_after+"(Black CASTLING)"]
        else:
            readableMove=[piece,pos_before,pos_after]
        if(capturedMove==True):
            readableMove[2]+="(Piece CAPTURED)"
        elif(piece=="pawn" and pieceNew!="pawn"):
            readableMove[2]+="(promotion: "+pieceNew+")"
        if(debug): print("DEBUG (functions (printMoves)): readableMove=",readableMove)
        readableMoveList.append(readableMove)
    if(noOutputMode==False):
        sortedLists=createPieceListforPrintOut(readableMoveList,colored,debug)
        print("INFO (functions (printMoves)): Moves for player",player,":")
        for list in sortedLists:
            print(list)

def inCheck(player, boardpositions, move, isEnPassant, debug):
    if(debug): print("(functions (inCheck)): move=",move)
    copiedBoardPositions=copy.deepcopy(boardpositions)
    moveInfo=copiedBoardPositions.playMove(player, move, [False,False], isEnPassant, False, debug)
    willPlayerBeInCheck=copiedBoardPositions.isPlayerInCheck(player,False)
    if(debug): print("(functions (inCheck)): willPlayerBeInCheck?",willPlayerBeInCheck,". If so remove move from list.")
    return willPlayerBeInCheck

def getValidMoves(player,boardpositions,moveCandidates,castlingCandidates,enPassantCandidates,debug=False,noOutputMode=False):
    moveID=0
    castlingLongID=-999
    castlingShortID=-999
    enPassantIDs=[-999,-999]
    outOfCheckMoves=[]
    for testMove in moveCandidates: # test normal moves
        if(debug and noOutputMode==False): print("(functions (nextMove)): testMove=",testMove)
        willPlayerBeInCheck=inCheck(player, boardpositions, testMove, False, debug)
        if(willPlayerBeInCheck==False):
            outOfCheckMoves.append(testMove)
            moveID+=1
    for castlingMove in castlingCandidates: # test castling moves
        if(debug and noOutputMode==False): print("DEBUG (functions (nextMove)): castlingMove=",castlingMove)
        startMove        =[castlingMove[0],castlingMove[1],castlingMove[2],[4,0],castlingMove[3]]
        if(castlingMove[3]==[2,0]): #castling long
            inBetweenMove=[castlingMove[0],castlingMove[1],castlingMove[2],[3,0],castlingMove[3]]
        elif(castlingMove[3]==[6,0]): #castling short
            inBetweenMove=[castlingMove[0],castlingMove[1],castlingMove[2],[5,0],castlingMove[3]]
        willPlayerBeInCheck1=inCheck(player, boardpositions, startMove    , False, debug)
        willPlayerBeInCheck2=inCheck(player, boardpositions, inBetweenMove, False, debug)
        willPlayerBeInCheck3=inCheck(player, boardpositions, castlingMove , False, debug)
        if(willPlayerBeInCheck1==False and willPlayerBeInCheck2==False and willPlayerBeInCheck3==False):
            if(castlingMove[3]==[2,0]):
                castlingLongID=moveID
            elif(castlingMove[3]==[6,0]):
                castlingShortID=moveID
            moveID+=1
            outOfCheckMoves.append(castlingMove)
    if(debug): print("INFO (functions (nextMove)): len(enPassantCandidates)=",len(enPassantCandidates))
    for num,enPassantMove in enumerate(enPassantCandidates): # test enPassant moves
        if(debug and noOutputMode==False): print("DEBUG (functions (nextMove)): enPassantMove=",enPassantMove)
        willPlayerBeInCheck=inCheck(player, boardpositions, enPassantMove, True, debug)
        if(willPlayerBeInCheck==False):
            enPassantIDs[num]=moveID
            moveID+=1
            outOfCheckMoves.append(enPassantMove)
    return outOfCheckMoves,castlingLongID,castlingShortID,enPassantIDs

def nextMove(gameMode,player,castling,enPassant,boardpositions,colored=False,debug=False,noOutputMode=False):
    finished=False
    playerInCheck=boardpositions.isPlayerInCheck(player,debug)
    if(playerInCheck==True and noOutputMode==False): print("Player",player,"is in CHECK!")
    ### GET VALID MOVES ###
    moveCandidates,castlingCandidates,enPassantCandidates=boardpositions.findAllowedMoves(player, castling[player], enPassant, debug)
    if(debug and noOutputMode==False):
        printMoves(player,moveCandidates,colored,noOutputMode,debug)
    outOfCheckMoves,castlingLongID,castlingShortID,enPassantIDs=getValidMoves(player,boardpositions,moveCandidates,castlingCandidates,enPassantCandidates,debug,noOutputMode)
    printMoves(player,outOfCheckMoves,colored,noOutputMode,debug)

    ### PLAY THE MOVE ###
    if(len(outOfCheckMoves)>0):
        if((gameMode=="WhiteVsComputer" and player==0) or (gameMode=="BlackVsComputer" and player==1) or gameMode=="WhiteVsBlack"):
            rand=int(input("Input your move:"))
            print("Choose move ID",rand)
            move=outOfCheckMoves[rand]
        elif(gameMode=="SelfplayNetworkVsRandom"):
            if(player==1): # random move
                rand=random.randint(0,len(outOfCheckMoves)-1)
                if(debug): print("(functions (nextMove)): rand=",rand)
                move=outOfCheckMoves[rand]
                if(noOutputMode==False): print("INFO (functions (nextMove): IDs: rand,castLong,castShort,enpassant=",rand,castlingLongID,castlingShortID,enPassantIDs)
            elif(player==0): # network move (no implemented yet!)
                rand=random.randint(0,len(outOfCheckMoves)-1)
                if(debug): print("(functions (nextMove)): rand=",rand)
                move=outOfCheckMoves[rand]
                if(noOutputMode==False): print("INFO (functions (nextMove): IDs: rand,castLong,castShort,enpassant=",rand,castlingLongID,castlingShortID,enPassantIDs)
        elif(gameMode=="SelfplayRandomVsRandom"):
            rand=random.randint(0,len(outOfCheckMoves)-1)
            if(debug): print("(functions (nextMove)): rand=",rand)
            move=outOfCheckMoves[rand]
            if(noOutputMode==False): print("INFO (functions (nextMove): IDs: rand,castLong,castShort,enpassant=",rand,castlingLongID,castlingShortID,enPassantIDs)
        else:
            time.sleep(1.5)
            rand=random.randint(0,len(outOfCheckMoves)-1)
            if(debug): print("(functions (nextMove)): rand=",rand)
            move=outOfCheckMoves[rand]
            if(noOutputMode==False): print("INFO (functions (nextMove): IDs: rand,castLong,castShort,enpassant=",rand,castlingLongID,castlingShortID,enPassantIDs)

        moveInfoList=[]
        if(rand==castlingLongID):
            if(noOutputMode==False): print("Player",player,"is CASTLING long!")
            castle=[True,False]
            moveInfoList=boardpositions.playMove(player, move, castle, False, True, debug)
            if(noOutputMode==False): print("INFO: Player",player,"is no longer allowed to castle.")
        elif(rand==castlingShortID):
            if(noOutputMode==False): print("Player",player,"is CASTLING short!")
            castle=[False,True]
            moveInfoList=boardpositions.playMove(player, move, castle, False, True, debug)
            if(noOutputMode==False): print("INFO: Player",player,"is no longer allowed to castle.")
        elif(rand==enPassantIDs[0] or rand==enPassantIDs[1]):
            castle=[False,False]
            moveInfoList=boardpositions.playMove(player, move, castle, True, True, debug)
            if(noOutputMode==False): print("INFO: Player",player,"just captured via en-passant at",move[2])
        else:
            castle=[False,False]
            moveInfoList=boardpositions.playMove(player, move, castle, False, True, debug)

        if(move[0]=="king" and move[2]==[4,0]):
            if(castling[player][0]==True or castling[player][1]==True):
                if(noOutputMode==False): print("INFO: Player",player,"is no longer allowed to castle.")
            castling[player]=[False,False]
        elif(move[0]=="rook" and move[2]==[0,0]):
            if(castling[player][0]==True):
                if(noOutputMode==False): print("INFO: Player",player,"is no longer allowed to castle long.")
            castling[player][0]=False
        elif(move[0]=="rook" and move[2]==[7,0]):
            if(castling[player][1]==True):
                if(noOutputMode==False): print("INFO: Player",player,"is no longer allowed to castle short.")
            castling[player][1]=False
        if(move[0]=="pawn" and move[1][1]==1 and move[2][1]==3):
           enPassant=[True,move[2]]
           print("INFO: Player",getOpponent(player),"can now take player",player,"'s pawn at",move[2],"via en-passant.")
        else:
            enPassant=[False,[-1,-1]]
        if(noOutputMode==False):
            boardpositions.printChessBoard(player,colored,moveInfoList,debug)
    elif(len(outOfCheckMoves)==0 and playerInCheck==True):
        if(noOutputMode==False): print("Player",player,"is CHECKMATE!")
        boardpositions.setWinner(getOpponent(player))
        finished=True
    elif(len(outOfCheckMoves)==0 and playerInCheck==False):
        if(noOutputMode==False): print("Player",player,"has no more moves available => Remis.")
        boardpositions.setWinner(-1)
        finished=True

    return castling,enPassant,finished
