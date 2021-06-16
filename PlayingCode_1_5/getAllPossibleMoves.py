import numpy as np
import moves
import numba as nb
from numba import types

moveDict=nb.typed.Dict.empty(
    key_type  =nb.types.UniTuple(nb.int64, 5),
    value_type=nb.int64)
moveList,moveDict=moves.getAllMoves(moveDict)

#num=758
#print("moveList[",num,"]=",moveList[num])
for i,move in enumerate(moveList):
    print("i,move,moveDict[",move,"]=",i,move,moveDict[move])

#t=[moveList[0][2],moveList[0][1]]
#print("t=",t)
