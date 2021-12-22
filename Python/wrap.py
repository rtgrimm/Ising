
import ctypes

import numpy as np
from past.types import long

from Ising import IntVector
from Python.Ising import vector_data


def PtrToArray(ptr, size):
    addr = long(ptr)

    return np.copy(np.ctypeslib.as_array(
        (ctypes.c_int32 * size).from_address(addr)))

def VecToMat(vec : IntVector):
    data = vector_data(vec)
    array = PtrToArray(data.ptr, data.size)

    return array
