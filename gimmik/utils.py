# -*- coding: utf-8 -*-

class BlockConfig(object):
    def __init__(self, thrd_gang_size, blk_dim, ufc_size, shr_size, precision, warp_size=32):
        self.thrd_gang_size = thrd_gang_size
        self.blk_dim = blk_dim
        self.ufc_size = ufc_size
        self.shr_size = shr_size
        self.precision = precision
        self.warp_size = warp_size

        self.warps = int(blk_dim/warp_size)
        self.active_threads = self.warps*thrd_gang_size*int(warp_size/thrd_gang_size)

        self.shr_var_size = int(shr_size/precision)
        self.shr_size_elem = int(thrd_gang_size*self.shr_var_size/self.active_threads)

        self.L1_size = int((ufc_size - shr_size)/precision)
        self.L1_per_thread = int(self.L1_size/self.active_threads)
        

def glb_addr(context, name, i, v, o):
    return f'{name}[SOA_IDX({i}, {v}) + {o}]'

def lcl_addr(name, v, o):
    return f'{name}[{v} + {o}]'

def read_glb_p(context, name, i, nv, o): # read global point
    csd = ', '.join(f'{name}[SOA_IDX({i}, {v}) + {o}]' for v in range(nv))
    return '{' + csd + '}'

def safe_src(func):
    def wrapper(*args, **kwargs):
        return f'({func(*args, **kwargs)})'
    return wrapper

def new_line(func):
    def wrapper(*args, **kwargs):
        return f'{func(*args, **kwargs)}\n'
    return wrapper

