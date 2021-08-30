import numpy as np
import moves
import functions
from functions import printInfo, printDebug, printError
from timeit import default_timer as timer
import pickle
import numba as nb
from numba.experimental import jitclass

############################################################################################################################
#### jited ChessBoard class ################################################################################################ 
############################################################################################################################

ChessBoardSpecs = [
    ('Board',         nb.types.Array(nb.types.int64, 2, 'C')),
    ('KingPositions', nb.types.Array(nb.types.int64, 2, 'C')), #nb.int64[:,:]),
    ('Castling',      nb.types.Array(nb.types.int64, 2, 'C')),
    ('EnPassant',     nb.types.int64),
    ('Reversed',      nb.boolean),
    ('NoOutputMode',  nb.boolean),
]

@jitclass(ChessBoardSpecs)
class ChessBoard(object):

    def __init__(self, noOutputMode) -> None:
        self.Board =         np.zeros((8,8), dtype = np.int64)
        self.KingPositions = np.array([[4,0],[4,0]], dtype = np.int64)
        self.Castling  =     np.array([[True, True], [True, True]], dtype = np.int64)
        self.EnPassant =     -1 # Enpassant colum. If -1, en-passant is not allowed in next move
        self.Reversed  =     False
        self.NoOutputMode = noOutputMode

    def printChessBoard(self, arg = "Board-"):
        lenChessBoard = len(self.Board)
        printInfo(self.NoOutputMode, "######", arg, "######\n## A B C D E F G H #")
        
        for row in range(lenChessBoard):
            rowInformation = []
            if self.Reversed == False:
                row = (lenChessBoard -1) - row
            rowInformation.append(str(row + 1))            
            for col in range(lenChessBoard):
                pieceUnicode = functions.getPieceUnicode(self.Board[row][col], False)
                rowInformation.append(str(pieceUnicode))
            rowInformation.append(" "+str(row + 1))
            if not self.NoOutputMode:
                print(" ".join(rowInformation))
            #printInfo("test", noOutputMode = self.NoOutputMode)

        printInfo(self.NoOutputMode, "# A B C D E F G H ##\n###### -Board ######")
            

############################################################################################################################
#### jited BoardManager class ############################################################################################## 
############################################################################################################################

BoardManagerSpecs = [
    ('ChessBoard'     , ChessBoard.class_type.instance_type),
    ('PieceBoards'    , nb.types.Array(nb.types.float64, 4, 'C')), #nb.int64[:,:,:,:]),
    ('Players'        , nb.types.DictType(nb.types.string, nb.types.int64)),
    ('Pieces'         , nb.types.DictType(nb.types.string, nb.types.int64)), #nb.types.ListType(nb.types.string)),
    ('CurrentPlayer'  , nb.types.string),
    ('CurrentOpponent', nb.types.string),
    ('PlyNumber'      , nb.int64),
    ('Finished'       , nb.boolean),
    ('NoOutputMode'   , nb.boolean),
]

@jitclass(BoardManagerSpecs)
class BoardManager(object):

    def __init__(self, chessBoard, noOutputMode) -> None:
        self.fillPlayerDict()
        self.fillPieceDict()
        self.PieceBoards =     np.zeros((2,6,8,8), dtype = np.float64)
        self.ChessBoard =      chessBoard
        self.PlyNumber =       1
        self.Finished =        False
        self.CurrentPlayer =   "white"
        self.CurrentOpponent = "black"        
        self.NoOutputMode =    noOutputMode

        
    def fillPlayerDict(self):
        self.Players = nb.typed.Dict.empty(key_type = nb.types.string, value_type = nb.types.int64)
        self.Players["white"] = 1
        self.Players["black"] = -1

        
    def fillPieceDict(self):
        self.Pieces = nb.typed.Dict.empty(key_type = nb.types.string, value_type = nb.types.int64)
        self.Pieces["king"]   = 1
        self.Pieces["queen"]  = 2
        self.Pieces["rook"]   = 3
        self.Pieces["bishop"] = 4
        self.Pieces["knight"] = 5
        self.Pieces["pawn"]   = 6

        
    def pawnInitializer(self) -> None:
        self.PieceBoards[0][5][1] = [1 for i in range(8)] #white
        self.PieceBoards[1][5][6] = [1 for i in range(8)] #black

        
    def knightInitializer(self) -> None:
        self.PieceBoards[0][4][0][1], self.PieceBoards[0][4][0][6] = 1, 1 #white
        self.PieceBoards[1][4][7][1], self.PieceBoards[1][4][7][6] = 1, 1 #black

        
    def bishopInitializer(self) -> None:
        self.PieceBoards[0][3][0][2], self.PieceBoards[0][3][0][5] = 1, 1 #white
        self.PieceBoards[1][3][7][2], self.PieceBoards[1][3][7][5] = 1, 1 #black        

        
    def rookInitializer(self) -> None:
        self.PieceBoards[0][2][0][0], self.PieceBoards[0][2][0][7] = 1, 1 #white
        self.PieceBoards[1][2][7][0], self.PieceBoards[1][2][7][7] = 1, 1 #black

        
    def queenInitializer(self) -> None:
        self.PieceBoards[0][1][0][3] = 1 #white
        self.PieceBoards[1][1][7][3] = 1 #black

        
    def kingInitializer(self) -> None:
        self.PieceBoards[0][0][0][4] = 1 #white
        self.PieceBoards[1][0][7][4] = 1 #black
        self.ChessBoard.KingPositions = np.array([[4,0],[4,0]], dtype = np.int64)
        self.ChessBoard.Castling = np.array([[True, True], [True, True]], dtype=np.int64) # [[white long,white short],[black long, black short]] still allowed?

        
    def initializePieceBoards(self) -> None:
        self.pawnInitializer()
        self.knightInitializer()
        self.bishopInitializer()
        self.rookInitializer()
        self.queenInitializer()
        self.kingInitializer()
        if __debug__:
            printDebug(inspect.stack()[0][3], "All PieceBoards initialized:", self.PieceBoards, cName = self.__class__.__name__)

            
    def setPieces(self) -> None:
        lenChessBoard = len(self.ChessBoard.Board)
        for playerIdx, player in enumerate(self.Players):
            playerSign = self.Players[player]
            for pieceIdx, piece in enumerate(self.Pieces):
                printInfo(self.NoOutputMode, "X", str(pieceIdx), str(piece))
                for row in range(lenChessBoard):
                    for col in range(lenChessBoard):
                        if(self.PieceBoards[playerIdx][pieceIdx][row][col] == 1):
                            self.ChessBoard.Board[row][col] = playerSign*(pieceIdx+1)
                            if piece == "king":
                                if player == self.CurrentPlayer:
                                    self.ChessBoard.KingPositions[0] = [col, row]
                                else:
                                    self.ChessBoard.KingPositions[1] = [col, (lenChessBoard - 1) - row]
        if self.CurrentPlayer == "black":
            self.ChessBoard.Reversed = True
        else:
            self.ChessBoard.Reversed = False
        if __debug__:
            printDebug(inspect.stack()[0][3], "All pieces placed onto ChessBoard:", self.ChessBoard.Board, cName = self.__class__.__name__)

        
    def setPositionFromFile(self, posFile) -> None:
        fileIn = open(posFile,"rb")
        self = pickle.load(fileIn)

        
    def writePositionToFile(self) -> None:
        inputID = self.getInputID(debug)
        fileOut = open("PositionsToPlayFrom/"+inputID+".pkl", "wb")
        pickle.dump([self],fileOut)
        fileOut.close()

        
    def getNormalMoves(self, mD, ownPositions, noOutputMode=False) -> nb.typed.List():
        normalMoves = nb.typed.List()
        onlyCaptureMoves = False
        for pieceIdx, piece in enumerate(self.Pieces):
            if __debug__:
                printDebug(inspect.stack()[0][3], "Current piece=",piece,"; ownPositions=",ownPositions[pieceIdx], cName = self.__class__.__name__)
            for piecePos in ownPositions[pieceIdx]:
                moveManager = moves.MoveManager(piece, piecePos, self.ChessBoard, self.CurrentPlayer)
                moveManager.setBasicPieceMoves(onlyCaptureMoves)
        return normalMoves
