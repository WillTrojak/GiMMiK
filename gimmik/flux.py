# -*- coding: utf-8 -*-

# import re

# from gimmik.utils import safe_src

# class FluxMapping(object):
#     def __init__(self, func, name, jac, ndims, v, **kwargs):
#         self.func = func
#         self.name = name
#         self.jac = jac
#         self.ndims = ndims
#         self.var = v

#         self.conflictable = locals()
#         self.conflictable.pop('func')

#         self.mapping, self.F = eval(f'{func}(name=name, ndims=ndims, v=v, **kwargs)')

#         if self._map_conflict():
#             raise ValueError('Gimmik: flux mapping conflict')

#     def _map_conflict(self):
#         for m in self.mapping:
#             for x in self.conflictable:
#                 if re.search(rf'\b({self.conflictable[x]})\b', m) != None:
#                     return True
#         return False

#     @safe_src
#     def _substitute(self, body, mapping):
#         body_o = ''
#         while body != body_o:
#             body_o = body
#             for m in mapping:
#                 body = re.sub(rf'\b({m})\b', f'{mapping[m]}', body)
#         return body

#     @safe_src
#     def build_src(self):

#         src = ''
#         for i in self.F:
#             sub_str = self._substitute(body=self.F[i], mapping=self.mapping)
#             src = src + f'({self.jac[i]})*{sub_str}' + ('', '+')[i < self.ndims-1]
#         self.src = src

#         return src


# def src(context, func, name, jac, ndims, v, **kwargs):
#     flux_map = FluxMapping(func, name, jac, ndims, v, **kwargs)

#     return flux_map.build_src()

# def euler(name, o, v, ndims):

#     if ndims == 2:
#         mapping = {'rho': f'{name}[0 + {o}]',
#                    'rhou': f'{name}[1 + {o}]',
#                    'rhov': f'{name}[2 + {o}]',
#                    'E': f'{name}[3 + {o}]',
#                    'P': '0.4*(E - 0.5*(rhou*rhou + rhov*rhov)/rho)',
#                    }

#         f = {0: {0: 'rhou', 1: 'rhov'},
#              1: {0: 'rhou*rhou/rho + P', 1: 'rhov*rhou/rho'},
#              2: {0: 'rhou*rhov/rho', 1: 'rhov*rhov/rho + P'},
#              3: {0: 'rhou*(E + P)/rho', 1: 'rhov*(E + P)/rho'}
#             }
#     elif ndims == 3:
#         mapping = {'rho': f'{name}[0 + {o}]',
#                    'rhou': f'{name}[1 + {o}]',
#                    'rhov': f'{name}[2 + {o}]',
#                    'rhow': f'{name}[3 + {o}]',
#                    'E': f'{name}[4 + {o}]',
#                    'P': '0.4*(E - 0.5*(rhou*rhou + rhov*rhov + rhow*rhow)/rho)',
#                    }
        
#         f = {0: {0: 'rhou', 1: 'rhov', 2: 'rhow'},
#              1: {0: 'rhou*rhou/rho + P', 1: 'rhov*rhou/rho', 2: 'rhow*rhou/rho'},
#              2: {0: 'rhou*rhov/rho', 1: 'rhov*rhov/rho + P', 2: 'rhow*rhov/rho'},
#              3: {0: 'rhou*rhow/rho', 1: 'rhov*rhow/rho', 2: 'rhow*rhow/rho + P'},
#              4: {0: 'rhou*(E + P)/rho', 1: 'rhov*(E + P)/rho', 2: 'rhow*(E + P)/rho'}
#             }

#     return mapping, f[v]


# def euler_global(name, o, v, i, ndims):

#     if ndims == 2:
#         mapping = {'rho':  f'{glb_addr(name, i, 0, o)}',
#                    'rhou': f'{glb_addr(name, i, 1, o)}',
#                    'rhov': f'{glb_addr(name, i, 2, o)}',
#                    'E':    f'{glb_addr(name, i, 3, o)}',
#                    'p': '0.4*(E - 0.5*(rhou*rhou + rhov*rhov)/rho)',
#                    }

#         f = {(0, 0): 'rhou',
#              (1, 0): 'rhou*rhou/rho + p',
#              (2, 0): 'rhou*rhov/rho',
#              (3, 0): 'rhou*(E + p)/rho',
#              (0, 1): 'rhov',
#              (1, 1): 'rhov*rhou/rho',
#              (2, 1): 'rhov*rhov/rho + p',
#              (3, 1): 'rhov*(E + p)/rho'
#             }
#     elif ndims == 3:
#         mapping = {'rho':  f'{glb_addr(name, i, 0, o)}',
#                    'rhou': f'{glb_addr(name, i, 1, o)}',
#                    'rhov': f'{glb_addr(name, i, 2, o)}',
#                    'rhow': f'{glb_addr(name, i, 3, o)}',
#                    'E':    f'{glb_addr(name, i, 4, o)}',
#                    'p': '0.4*(E - 0.5*(rhou*rhou + rhov*rhov + rhow*rhow)/rho)',
#                    }
        
#         f = {(0, 0): 'rhou',
#              (1, 0): 'rhou*rhou/rho + p',
#              (2, 0): 'rhou*rhov/rho',
#              (3, 0): 'rhou*rhow/rho',
#              (4, 0): 'rhou*(E + p)/rho',
#              (0, 1): 'rhov',
#              (1, 1): 'rhov*rhou/rho',
#              (2, 1): 'rhov*rhov/rho + p',
#              (3, 1): 'rhov*rhow/rho',
#              (4, 1): 'rhov*(E + p)/rho',
#              (0, 2): 'rhow',
#              (1, 2): 'rhow*rhou/rho',
#              (2, 2): 'rhow*rhov/rho',
#              (3, 2): 'rhow*rhow/rho + p',
#              (4, 2): 'rhow*(E + p)/rho',
#             }

#     return mapping, f[v]

# def linadv_n(name, o, v, ndims):
    
#     mapping = {f'phi{v}': f'{name}[{v} + {o}]'}

#     f_sub = dict()
#     for i in range(ndims):
#         f_sub[i] = f'phi{v}'

#     return mapping, f_sub

# def linadv5(name, o, v, ndims):
    
#     mapping = {'phi1': f'{name}[0 + {o}]',
#                'phi2': f'{name}[1 + {o}]',
#                'phi3': f'{name}[2 + {o}]',
#                'phi4': f'{name}[3 + {o}]',
#                'phi5': f'{name}[4 + {o}]'
#               }

#     f = {0: 'phi1',
#          1: 'phi2',
#          2: 'phi3',
#          3: 'phi4',
#          4: 'phi5'
#         }

#     return mapping, f[v]

# def linadv5_glb(name, o, v, i, ndims):
    
#     mapping = {'phi1': f'{glb_addr(name, i, 0, o)}',
#                'phi2': f'{glb_addr(name, i, 1, o)}',
#                'phi3': f'{glb_addr(name, i, 2, o)}',
#                'phi4': f'{glb_addr(name, i, 3, o)}',
#                'phi5': f'{glb_addr(name, i, 4, o)}',
#               }

#     f = {0: 'phi1',
#          1: 'phi2',
#          2: 'phi3',
#          3: 'phi4',
#          4: 'phi5'
#         }

#     return mapping, f[v]

# def linadv2_global(name, o, v, i, ndims):
    
#     mapping = {'phi1': f'{glb_addr(name, i, 0, o)}',
#                'phi2': f'{glb_addr(name, i, 1, o)}',
#               }

#     f = {0: 'phi1',
#          1: 'phi2'
#         }

#     return mapping, f[v]
