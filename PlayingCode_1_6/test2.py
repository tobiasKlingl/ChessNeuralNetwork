import numpy as np
from numba.types import string, int32, float32, ListType
from numba import typeof, typed, njit
from numba.typed import List
from numba.experimental import jitclass

Dict = {"white": np.array([4, 0]), "black": np.array([4, 0])}

b = Dict


b["white"] = np.array([6, 3])


#print(Dict[0])
print(Dict)
print(b)
