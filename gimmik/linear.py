# -*- coding: utf-8 -*-

from gimmik.fluxcls import BaseFlux

class LinearFlux(BaseFlux):
    def __init__(self, ndims, fargs):
        super(LinearFlux, self).__init__(ndims, fargs)

    def _flux(self, q, v):
        c = self.fargs
        a = c['a']

        mapping = {f'phi{v}': q[v]}

        f_sub = dict()
        for i in range(c['ndims']):
            f_sub[i] = f'{a[i]}*phi{v}'

        return mapping, f_sub



def linadv_n(name, macro, ndims, v, sub):
    
    mapping = {f'phi{v}': f'{name}[{macro}({v}{sub})]'}

    f_sub = dict()
    for i in range(ndims):
        f_sub[i] = f'phi{v}'

    return mapping, f_sub

def linadv5(name, o, v, ndims):
    
    mapping = {'phi1': f'{name}[0 + {o}]',
               'phi2': f'{name}[1 + {o}]',
               'phi3': f'{name}[2 + {o}]',
               'phi4': f'{name}[3 + {o}]',
               'phi5': f'{name}[4 + {o}]'
              }

    f = {0: 'phi1',
         1: 'phi2',
         2: 'phi3',
         3: 'phi4',
         4: 'phi5'
        }

    return mapping, f[v]

def linadv5_glb(name, o, v, i, ndims):
    
    mapping = {'phi1': f'{glb_addr(name, i, 0, o)}',
               'phi2': f'{glb_addr(name, i, 1, o)}',
               'phi3': f'{glb_addr(name, i, 2, o)}',
               'phi4': f'{glb_addr(name, i, 3, o)}',
               'phi5': f'{glb_addr(name, i, 4, o)}',
              }

    f = {0: 'phi1',
         1: 'phi2',
         2: 'phi3',
         3: 'phi4',
         4: 'phi5'
        }

    return mapping, f[v]

def linadv2_global(name, o, v, i, ndims):
    
    mapping = {'phi1': f'{glb_addr(name, i, 0, o)}',
               'phi2': f'{glb_addr(name, i, 1, o)}',
              }

    f = {0: 'phi1',
         1: 'phi2'
        }

    return mapping, f[v]
