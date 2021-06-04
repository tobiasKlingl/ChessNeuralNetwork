import numpy as np

class ChessBoard(object):
    def __init__(self, color="", piece="", num_rows=8, num_columns=8, num_pieces=6):
        self.Color = color
        self.Piece = piece
        self.pieceBoardMatrix = np.zeros(num_rows * num_columns)
        self.pieceBoardMatrix.shape = (num_rows, num_columns)                                        
 
