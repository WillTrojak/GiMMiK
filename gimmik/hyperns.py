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

            s = {0: '0.',
                 1: '0.',
                 2: '0.',
                 3: f'-qx*{rtr}',
                 4: f'-qy*{rtr}',
                 5: f'-rx*{rtr}',
                 6: f'-ry*{rtr}',
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

            s = { 0: '0.',
                  1: '0.',
                  2: '0.',
                  3: '0.',
                  4: f'-qx*{rtr}',
                  5: f'-qy*{rtr}',
                  6: f'-qz*{rtr}',
                  7: f'-rx*{rtr}',
                  8: f'-ry*{rtr}',
                  9: f'-rz*{rtr}',
                 10: f'-sx*{rtr}',
                 11: f'-sy*{rtr}',
                 12: f'-sz*{rtr}',
                }

        return mapping, f[v], s[v]


def hyper(name, macro, ndims, v, sub):
    ac_zeta = 2.5
    nu = 6.25e-4
    tr = 5e-3
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

        s = {0: '0.',
             1: '0.',
             2: '0.',
             3: f'-qx*{rtr}',
             4: f'-qy*{rtr}',
             5: f'-rx*{rtr}',
             6: f'-ry*{rtr}',
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

        s = { 0: '0.',
              1: '0.',
              2: '0.',
              3: '0.',
              4: f'-qx*{rtr}',
              5: f'-qy*{rtr}',
              6: f'-qz*{rtr}',
              7: f'-rx*{rtr}',
              8: f'-ry*{rtr}',
              9: f'-rz*{rtr}',
             10: f'-sx*{rtr}',
             11: f'-sy*{rtr}',
             12: f'-sz*{rtr}',
            }

    return mapping, f[v], s[v]

def acmhd_24_acc(context, v, s, d, jac, i, j, k):
    flux = {0: f'{d}*({jac}[0]*({s}[SHR_IDX_I(el,0,{i},{j},{k})]*{s}[SHR_IDX_I(el,0,{i},{j},{k})] + {s}[SHR_IDX_I(el,3,{i},{j},{k})]) + {jac}[1]*({s}[SHR_IDX_I(el,0,{i},{j},{k})]*{s}[SHR_IDX_I(el,1,{i},{j},{k})] + {s}[SHR_IDX_I(el,4,{i},{j},{k})]) + {jac}[0]*({s}[SHR_IDX_I(el,0,{i},{j},{k})]*{s}[SHR_IDX_I(el,2,{i},{j},{k})] + {s}[SHR_IDX_I(el,5,{i},{j},{k})]))',
            1: f'{d}*({jac}[0]*({s}[SHR_IDX_I(el,1,{i},{j},{k})]*{s}[SHR_IDX_I(el,0,{i},{j},{k})] + {s}[SHR_IDX_I(el,6,{i},{j},{k})]) + {jac}[1]*({s}[SHR_IDX_I(el,1,{i},{j},{k})]*{s}[SHR_IDX_I(el,1,{i},{j},{k})] + {s}[SHR_IDX_I(el,7,{i},{j},{k})]) + {jac}[0]*({s}[SHR_IDX_I(el,1,{i},{j},{k})]*{s}[SHR_IDX_I(el,2,{i},{j},{k})] + {s}[SHR_IDX_I(el,8,{i},{j},{k})]))',
            2: f'{d}*({jac}[0]*({s}[SHR_IDX_I(el,2,{i},{j},{k})]*{s}[SHR_IDX_I(el,0,{i},{j},{k})] + {s}[SHR_IDX_I(el,9,{i},{j},{k})]) + {jac}[1]*({s}[SHR_IDX_I(el,2,{i},{j},{k})]*{s}[SHR_IDX_I(el,1,{i},{j},{k})] + {s}[SHR_IDX_I(el,10,{i},{j},{k})]) + {jac}[0]*({s}[SHR_IDX_I(el,2,{i},{j},{k})]*{s}[SHR_IDX_I(el,2,{i},{j},{k})] + {s}[SHR_IDX_I(el,11,{i},{j},{k})]))',
            3:  f'{d}*{jac}[0]*{s}[SHR_IDX_I(el,0,{i},{j},{k})]',
            4:  f'{d}*{jac}[1]*{s}[SHR_IDX_I(el,0,{i},{j},{k})]',
            5:  f'{d}*{jac}[2]*{s}[SHR_IDX_I(el,0,{i},{j},{k})]',
            6:  f'{d}*{jac}[0]*{s}[SHR_IDX_I(el,1,{i},{j},{k})]',
            7:  f'{d}*{jac}[1]*{s}[SHR_IDX_I(el,1,{i},{j},{k})]',
            8:  f'{d}*{jac}[2]*{s}[SHR_IDX_I(el,1,{i},{j},{k})]',
            9:  f'{d}*{jac}[0]*{s}[SHR_IDX_I(el,2,{i},{j},{k})]',
            10: f'{d}*{jac}[1]*{s}[SHR_IDX_I(el,2,{i},{j},{k})]',
            11: f'{d}*{jac}[2]*{s}[SHR_IDX_I(el,2,{i},{j},{k})]',
           }
    return flux[v]

def acmhd_24_import(context, v, g, ldg, r, s, c, p, i, j, k):
    imp = {0: f'{r}[0] = __ldg({g} + SOA_IDX(eg,0) + ({i} + {j}*{p} + {k}*{p*p})*{ldg});',
           1: f'{r}[1] = __ldg({g} + SOA_IDX(eg,1) + ({i} + {j}*{p} + {k}*{p*p})*{ldg});\n{s}[SHR_IDX_I(el,0,{i},{j},{k})] = {r}[1];',
           2: f'{r}[2] = __ldg({g} + SOA_IDX(eg,2) + ({i} + {j}*{p} + {k}*{p*p})*{ldg});\n{s}[SHR_IDX_I(el,1,{i},{j},{k})] = {r}[2];',
           3: f'{r}[3] = __ldg({g} + SOA_IDX(eg,3) + ({i} + {j}*{p} + {k}*{p*p})*{ldg});\n{s}[SHR_IDX_I(el,2,{i},{j},{k})] = {r}[3];',
           4: f"{r}[4] = __ldg({g} + SOA_IDX(eg,4) + ({i} + {j}*{p} + {k}*{p*p})*{ldg});\n{s}[SHR_IDX_I(el,3,{i},{j},{k})] = {-c['nu']}*{r}[4] + {r}[0];",
           5: f"{r}[5] = __ldg({g} + SOA_IDX(eg,5) + ({i} + {j}*{p} + {k}*{p*p})*{ldg});\n{s}[SHR_IDX_I(el,4,{i},{j},{k})] = {-c['nu']}*{r}[5];",
           6: f"{r}[6] = __ldg({g} + SOA_IDX(eg,6) + ({i} + {j}*{p} + {k}*{p*p})*{ldg});\n{s}[SHR_IDX_I(el,5,{i},{j},{k})] = {-c['nu']}*{r}[6];",
           7: f"{r}[7] = __ldg({g} + SOA_IDX(eg,7) + ({i} + {j}*{p} + {k}*{p*p})*{ldg});\n{s}[SHR_IDX_I(el,6,{i},{j},{k})] = {-c['nu']}*{r}[7];",
           8: f"{r}[8] = __ldg({g} + SOA_IDX(eg,8) + ({i} + {j}*{p} + {k}*{p*p})*{ldg});\n{s}[SHR_IDX_I(el,7,{i},{j},{k})] = {-c['nu']}*{r}[8] + {r}[0];",
           9: f"{r}[9] = __ldg({g} + SOA_IDX(eg,9) + ({i} + {j}*{p} + {k}*{p*p})*{ldg});\n{s}[SHR_IDX_I(el,8,{i},{j},{k})] = {-c['nu']}*{r}[9];",
           10: f"{r}[10] = __ldg({g} + SOA_IDX(eg,10) + ({i} + {j}*{p} + {k}*{p*p})*{ldg});\n{s}[SHR_IDX_I(el,9,{i},{j},{k})] = {-c['nu']}*{r}[10];",
           11: f"{r}[11] = __ldg({g} + SOA_IDX(eg,11) + ({i} + {j}*{p} + {k}*{p*p})*{ldg});\n{s}[SHR_IDX_I(el,10,{i},{j},{k})] = {-c['nu']}*{r}[11];",
           12: f"{r}[12] = __ldg({g} + SOA_IDX(eg,12) + ({i} + {j}*{p} + {k}*{p*p})*{ldg});\n{s}[SHR_IDX_I(el,11,{i},{j},{k})] = {-c['nu']}*{r}[12] + {r}[0];",
          }
    return imp[v]