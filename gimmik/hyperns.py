# -*- coding: utf-8 -*-

from gimmik.fluxcls import BaseFlux

class HyperNSFlux(BaseFlux):
    def __init__(self, ndims, fargs):
        super(HyperNSFlux, self).__init__(ndims, fargs)

    def _flux(self, q, v):
        c = self.fargs

        ac_zeta = c['ac-zeta']
        nu = c['nu']
        tr = c['tr']
        rtr = 1/tr

        if self.ndims == 2:
            mapping = {'p': q[0],
                       'u': q[1],
                       'v': q[2],
                       'qx': q[3],
                       'qy': q[4],
                       'rx': q[5],
                       'ry': q[6],
                       }
            
            f = { 0: {0: f'{ac_zeta}*u', 1: f'{ac_zeta}*v'},
                  1: {0: f'u*u + p - {nu}*qx', 1: f'v*u - {nu}*qy'},
                  2: {0: f'u*v - {nu}*rx',     1: f'v*v + p - {nu}*ry'},
                  3: {0: f'-{rtr}*u'},
                  4: {1: f'-{rtr}*u'},
                  5: {0: f'-{rtr}*v'},
                  6: {1: f'-{rtr}*v'},
                 }

        elif self.ndims == 3:
            mapping = {'p': q[0],
                       'u': q[1],
                       'v': q[2],
                       'w': q[3],
                       'qx': q[4],
                       'qy': q[5],
                       'qz': q[6],
                       'rx': q[7],
                       'ry': q[8],
                       'rz': q[9],
                       'sx': q[10],
                       'sy': q[11],
                       'sz': q[12],
                        }

            f = { 0: {0: f'{ac_zeta}*u', 1: f'{ac_zeta}*v', 2: f'{ac_zeta}*w'},
                  1: {0: f'u*u + p - {nu}*qx', 1: f'v*u - {nu}*qy',     2: f'w*u - {nu}*qz'},
                  2: {0: f'u*v - {nu}*rx',     1: f'v*v + p - {nu}*ry', 2: f'w*v - {nu}*rz'},
                  3: {0: f'u*w - {nu}*sx',     1: f'v*w - {nu}*sy',     2: f'w*w + p - {nu}*sz'},
                  4: {0: f'-{rtr}*u'},
                  5: {1: f'-{rtr}*u'},
                  6: {2: f'-{rtr}*u'},
                  7: {0: f'-{rtr}*v'},
                  8: {1: f'-{rtr}*v'},
                  9: {2: f'-{rtr}*v'},
                 10: {0: f'-{rtr}*w'},
                 11: {1: f'-{rtr}*w'},
                 12: {2: f'-{rtr}*w'},
                 }

        return mapping, f[v]


def hyper(name, macro, ndims, v, sub):
    ac_zeta = 2.5
    nu = 6.25e-4
    tr = 0.001
    rtr = 1/tr

    if ndims == 2:
        mapping = {'p': f'{name}[{macro}(0{sub})]',
                   'u': f'{name}[{macro}(1{sub})]',
                   'v': f'{name}[{macro}(2{sub})]',
                   'qx': f'{name}[{macro}(4{sub})]',
                   'qy': f'{name}[{macro}(5{sub})]',
                   'rx': f'{name}[{macro}(7{sub})]',
                   'ry': f'{name}[{macro}(8{sub})]',
                   }
        
        f = { 0: {0: f'{ac_zeta}*u', 1: f'{ac_zeta}*v'},
              1: {0: f'u*u + p - {nu}*qx', 1: f'v*u - {nu}*qy'},
              2: {0: f'u*v - {nu}*rx',     1: f'v*v + p - {nu}*ry'},
              3: {0: f'-{rtr}*u'},
              4: {1: f'-{rtr}*u'},
              5: {0: f'-{rtr}*v'},
              6: {1: f'-{rtr}*v'},
             }

    elif ndims == 3:
        mapping = {'p': f'{name}[{macro}(0{sub})]',
                   'u': f'{name}[{macro}(1{sub})]',
                   'v': f'{name}[{macro}(2{sub})]',
                   'w': f'{name}[{macro}(3{sub})]',
                   'qx': f'{name}[{macro}(4{sub})]',
                   'qy': f'{name}[{macro}(5{sub})]',
                   'qz': f'{name}[{macro}(6{sub})]',
                   'rx': f'{name}[{macro}(7{sub})]',
                   'ry': f'{name}[{macro}(8{sub})]',
                   'rz': f'{name}[{macro}(9{sub})]',
                   'sx': f'{name}[{macro}(10{sub})]',
                   'sy': f'{name}[{macro}(11{sub})]',
                   'sz': f'{name}[{macro}(12{sub})]',
                    }

        f = { 0: {0: f'{ac_zeta}*u', 1: f'{ac_zeta}*v', 2: f'{ac_zeta}*w'},
              1: {0: f'u*u + p - {nu}*qx', 1: f'v*u - {nu}*qy',     2: f'w*u - {nu}*qz'},
              2: {0: f'u*v - {nu}*rx',     1: f'v*v + p - {nu}*ry', 2: f'w*v - {nu}*rz'},
              3: {0: f'u*w - {nu}*sx',     1: f'v*w - {nu}*sy',     2: f'w*w + p - {nu}*sz'},
              4: {0: f'-{rtr}*u'},
              5: {1: f'-{rtr}*u'},
              6: {2: f'-{rtr}*u'},
              7: {0: f'-{rtr}*v'},
              8: {1: f'-{rtr}*v'},
              9: {2: f'-{rtr}*v'},
             10: {0: f'-{rtr}*w'},
             11: {1: f'-{rtr}*w'},
             12: {2: f'-{rtr}*w'},
             }

    return mapping, f[v]