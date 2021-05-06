# -*- coding: utf-8 -*-

from gimmik.manager import BaseManager
from gimmik.methods import Plane3d_shr, Plane3d
from gimmik.utils import BlockConfig, safe_src

def plane_method_3d(context, A, p, nvars, ndims, flux_func, block_config, mem,
                    thrd_v, fargs, opargs):

    method = Plane3d(block_config, A, p, nvars, ndims, thrd_v, mem, flux_func, fargs)

    return method.build(ld_opt=opargs['ld_opt'], st_opt=opargs['st_opt'],
                        intl_opt=opargs['intl_opt'],
                        mem_debug=opargs['mem_debug'],
                       )