# -*- coding: utf-8 -*-

import numpy as np
import re


def generator(context, mat, beta, shr_name, shr_os, shr_size, split, rep=1):
    src = ''

    split = SplitMatrix(mat, beta, shr_name, shr_os, shr_size, split, rep)
    src += split.build()

    return src


class SplitMatrix(object):
    def __init__(self, mat, beta, shr_name, shr_os, shr_size, split, rep):
        super().__init__()

        self.mat = mat
        (self.n, self.m) = np.shape(mat)
        self.beta = beta

        self.shr_name = shr_name
        self.shr_os = shr_os
        self.shr_size = shr_size

        self.split = split
        self.rep = rep

    def _init_source(self):
        src_list = []
        for j, jx in enumerate(self.mat):
            src_list.append(f"dotp = {' + '.join('{kx}*b[i + {k}*ldb]'.format(k=k, kx=kx) for k, kx in enumerate(jx) if kx != 0) or 0};\n")

            if self.beta == 0:
                src_list.append(f'__stcg(c + i + {j}*ldc, dotp);\n')
            elif self.beta == 1:
                src_list.append(f'c[i + {j}*ldc] += dotp;\n')
            else:
                src_list.append(f'c[i + {j}*ldc] = dotp + {self.beta}*c[i + {j}*ldc];\n')

        self.src_list = src_list

    def _shared_load(self):
        src_list = []

        self.shr_sub = {}

        points_per_warp = int(self.m/self.split)
        for w in range(self.split):
            src_block = []
            src_block.append(f'if (warp == {w}){{\n')
            for j in range(points_per_warp):
                shr_var = f'{self.shr_name}[{self.shr_os} + {w*points_per_warp + j}]'
                glb_var = f'b[i + {w*points_per_warp + j}*ldb]'

                self.shr_sub[glb_var] = shr_var
                src_block.append(f'{shr_var} = {glb_var};\n')

            src_block.append('}\n')
            src_list += src_block
        
        remainder = self.m - self.split*points_per_warp
        for w in range(remainder):
            shr_var = f'{self.shr_name}[{self.shr_os} + {self.split*points_per_warp + w}]'
            glb_var = f'b[i + {self.split*points_per_warp + w}*ldb]'
            src_list.append(f'if( warp == {w})\n {shr_var}={glb_var};\n')
            self.shr_sub[glb_var] = shr_var

        src_list.append('__syncthreads();\n')
        self.shr_block = src_list

    def _use_shared(self):
        for i in range(len(self.src_list)):
            for v in re.findall(r'b\[i\s?\+\s?[0-9]*\*ldb\]', self.src_list[i]):
                self.src_list[i] = self.src_list[i].replace(v, self.shr_sub[v])

    def _split_comp_block(self):
        src = ''
        n = len(self.src_list)
        lines_per_warp = int(n/self.split)

        w = 0
        for i, s in enumerate(self.src_list):
            if i % lines_per_warp == 0 and w != self.split:
                src += f'if (warp == {w}){{ \n'
                w += 1

            src += s
        
            if i % lines_per_warp == lines_per_warp - 1 and w != self.split:
                src += '}\n'
        src += '}\n'
        return src

    def build(self):
        self._init_source()
        self._shared_load()
        self._use_shared()

        src = ''
        for s in (self.shr_block):
            src += s
        src += self._split_comp_block()

        return src
