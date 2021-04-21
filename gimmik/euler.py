# -*- coding: utf-8 -*-

import re

from gimmik.fluxcls import BaseFlux
from gimmik.utils import safe_src

class EulerFlux(BaseFlux):
    def __init__(self, ndims, fargs):
        super(EulerFlux, self).__init__(ndims, fargs)

    def _flux(self, q, v):
        c = self.fargs
        gamma = c['gamma']

        if self.ndims == 2:
            mapping = {'rho': q[0],
                       'rhou': q[1],
                       'rhov': q[2],
                       'E': q[3],
                       'P': f'{gamma-1}*(E - 0.5*(rhou*rhou + rhov*rhov)/rho)',
                       }

            f = {0: {0: 'rhou', 1: 'rhov'},
                 1: {0: 'rhou*rhou/rho + P', 1: 'rhov*rhou/rho'},
                 2: {0: 'rhou*rhov/rho', 1: 'rhov*rhov/rho + P'},
                 3: {0: 'rhou*(E + P)/rho', 1: 'rhov*(E + P)/rho'}
                }
        elif self.ndims == 3:
            mapping = {'rho': q[0],
                       'rhou': q[1],
                       'rhov': q[2],
                       'rhow': q[3],
                       'E': q[4],
                       'P': f'{gamma-1}*(E - 0.5*(rhou*rhou + rhov*rhov + rhow*rhow)/rho)',
                       }
            
            f = {0: {0: 'rhou', 1: 'rhov', 2: 'rhow'},
                 1: {0: 'rhou*rhou/rho + P', 1: 'rhov*rhou/rho', 2: 'rhow*rhou/rho'},
                 2: {0: 'rhou*rhov/rho', 1: 'rhov*rhov/rho + P', 2: 'rhow*rhov/rho'},
                 3: {0: 'rhou*rhow/rho', 1: 'rhov*rhow/rho', 2: 'rhow*rhow/rho + P'},
                 4: {0: 'rhou*(E + P)/rho', 1: 'rhov*(E + P)/rho', 2: 'rhow*(E + P)/rho'}
                }

        return mapping, f[v]


def euler_l(name, o, v, ndims):

    if ndims == 2:
        mapping = {'rho': f'{name}[0 + {o}]',
                   'rhou': f'{name}[1 + {o}]',
                   'rhov': f'{name}[2 + {o}]',
                   'E': f'{name}[3 + {o}]',
                   'P': '0.4*(E - 0.5*(rhou*rhou + rhov*rhov)/rho)',
                   }

        f = {0: {0: 'rhou', 1: 'rhov'},
             1: {0: 'rhou*rhou/rho + P', 1: 'rhov*rhou/rho'},
             2: {0: 'rhou*rhov/rho', 1: 'rhov*rhov/rho + P'},
             3: {0: 'rhou*(E + P)/rho', 1: 'rhov*(E + P)/rho'}
            }
    elif ndims == 3:
        mapping = {'rho': f'{name}[0 + {o}]',
                   'rhou': f'{name}[1 + {o}]',
                   'rhov': f'{name}[2 + {o}]',
                   'rhow': f'{name}[3 + {o}]',
                   'E': f'{name}[4 + {o}]',
                   'P': '0.4*(E - 0.5*(rhou*rhou + rhov*rhov + rhow*rhow)/rho)',
                   }
        
        f = {0: {0: 'rhou', 1: 'rhov', 2: 'rhow'},
             1: {0: 'rhou*rhou/rho + P', 1: 'rhov*rhou/rho', 2: 'rhow*rhou/rho'},
             2: {0: 'rhou*rhov/rho', 1: 'rhov*rhov/rho + P', 2: 'rhow*rhov/rho'},
             3: {0: 'rhou*rhow/rho', 1: 'rhov*rhow/rho', 2: 'rhow*rhow/rho + P'},
             4: {0: 'rhou*(E + P)/rho', 1: 'rhov*(E + P)/rho', 2: 'rhow*(E + P)/rho'}
            }

    return mapping, f[v]


def euler_global(name, o, v, i, ndims):

    if ndims == 2:
        mapping = {'rho':  f'{glb_addr(name, i, 0, o)}',
                   'rhou': f'{glb_addr(name, i, 1, o)}',
                   'rhov': f'{glb_addr(name, i, 2, o)}',
                   'E':    f'{glb_addr(name, i, 3, o)}',
                   'p': '0.4*(E - 0.5*(rhou*rhou + rhov*rhov)/rho)',
                   }

        f = {(0, 0): 'rhou',
             (1, 0): 'rhou*rhou/rho + p',
             (2, 0): 'rhou*rhov/rho',
             (3, 0): 'rhou*(E + p)/rho',
             (0, 1): 'rhov',
             (1, 1): 'rhov*rhou/rho',
             (2, 1): 'rhov*rhov/rho + p',
             (3, 1): 'rhov*(E + p)/rho'
            }
    elif ndims == 3:
        mapping = {'rho':  f'{glb_addr(name, i, 0, o)}',
                   'rhou': f'{glb_addr(name, i, 1, o)}',
                   'rhov': f'{glb_addr(name, i, 2, o)}',
                   'rhow': f'{glb_addr(name, i, 3, o)}',
                   'E':    f'{glb_addr(name, i, 4, o)}',
                   'p': '0.4*(E - 0.5*(rhou*rhou + rhov*rhov + rhow*rhow)/rho)',
                   }
        
        f = {(0, 0): 'rhou',
             (1, 0): 'rhou*rhou/rho + p',
             (2, 0): 'rhou*rhov/rho',
             (3, 0): 'rhou*rhow/rho',
             (4, 0): 'rhou*(E + p)/rho',
             (0, 1): 'rhov',
             (1, 1): 'rhov*rhou/rho',
             (2, 1): 'rhov*rhov/rho + p',
             (3, 1): 'rhov*rhow/rho',
             (4, 1): 'rhov*(E + p)/rho',
             (0, 2): 'rhow',
             (1, 2): 'rhow*rhou/rho',
             (2, 2): 'rhow*rhov/rho',
             (3, 2): 'rhow*rhow/rho + p',
             (4, 2): 'rhow*(E + p)/rho',
            }

    return mapping, f[v]