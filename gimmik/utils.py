# -*- coding: utf-8 -*-

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

