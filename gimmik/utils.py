# -*- coding: utf-8 -*-

class BlockConfig(object):
    def __init__(self, opargs, thrd_gang_size, blk_dim, ufc_size, shr_size, precision, warp_size=32):
        self.opargs = opargs
        self.thrd_gang_size = thrd_gang_size
        self.blk_dim = blk_dim
        self.ufc_size = ufc_size
        self.shr_size = shr_size
        self.precision = precision
        self.warp_size = warp_size

        self.warps = int(blk_dim/warp_size)
        self.active_threads = self.warps*thrd_gang_size*int(warp_size/thrd_gang_size)
        self.elem_warp_max = int(warp_size/thrd_gang_size)

        self.shr_var_size = int(8*shr_size/precision)
        self.shr_size_elem = int(thrd_gang_size*self.shr_var_size/self.active_threads)
        if opargs['shr_bdc']:
            self.usable_shr_elem = self.shr_size_elem - int(warp_size)
        else:
            self.usable_shr_elem = self.shr_size_elem

        self.shr_offset_rem = int(self.shr_size_elem % warp_size)

        #self.thrd_gang_size*(self.elem_warp_max-1)

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

