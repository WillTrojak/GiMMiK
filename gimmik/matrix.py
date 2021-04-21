# -*- coding: utf-8 -*-

import numpy as np

from gimmik.utils import new_line, safe_src

def c_register(context, A, name='mat'):
    A.name = name
    (A.n, A.m) = np.shape(A.mat)

    source = f'{A.mtype} {name}[{A.n*A.m}] = '
    source += '{'

    for j in range(A.m):
        for i in range(A.n):
            source += f'{A.mat[i,j]},'

    return source[0:-1] + '}'

class GimmikMatrix(object):
    def __init__(self, mat, mtype):
        self.mat = mat.copy()
        self.mtype = mtype

    @safe_src
    def matrix_value(self, i, j):
        if isinstance(i, str) and isinstance(j, str):
            return f'{self.name}[{i} + ({j})*{self.n}]'
        elif isinstance(i,str):
            return f'{self.name}[{i} + {j*self.n}]'
        elif isinstance(j,str):
            return f'{self.name}[{i} + ({j})*{self.n}]'
        else:
            return f'{self.mat[i,j]}'
        