# -*- coding: utf-8 -*-

from gimmik.manager import BaseManager
from gimmik.memory import GlobalMemory, LocalMemory, SharedMemory, RegisterMemory
from gimmik.methods import Plane3d, Planar3dMemoryManger
from gimmik.utils import safe_src

def plane_method_3d(context, A, elem, p, nvars, flux_func, shr_size, thrd_v, ldi, ldo, 
                    macro_name, shr_offset, fargs):

    glb_mem = GlobalMemory('b', elem, p, nvars, thrd_v, ldi, macro_name)
    glb_out_mem = GlobalMemory('c', elem, p, nvars, thrd_v, ldo, macro_name)
    shr_mem = SharedMemory('bsd', p, nvars, shr_offset, thrd_v, size=shr_size)
    #shrd_mem = SharedMemory('bsd', p, nvars, shr_offset, thrd_v, size=shrd_size)
    lcly_mem = LocalMemory('bly', p, nvars, size=p*nvars)
    lclx_mem = LocalMemory('blx', p, nvars, size=nvars)
    acc_reg = RegisterMemory('accl', size=nvars, rid='acc')


    mem = Planar3dMemoryManger(glb_mem, glb_out_mem, shr_mem, lcly_mem, lclx_mem, acc_reg)

    ndims = 3
    fargs['ac-zeta'] = 2.5
    fargs['nu'] = 1e-3
    fargs['tr'] = 1e-3
    method = Plane3d(A, p, nvars, ndims, thrd_v, mem, flux_func, fargs)

    return method.build()