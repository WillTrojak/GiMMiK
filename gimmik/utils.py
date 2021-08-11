# -*- coding: utf-8 -*-

import numpy as np

def nrows(context, mat):
    return int(np.shape(mat)[0])

def ncols(context, mat):
    return int(np.shape(mat)[1])