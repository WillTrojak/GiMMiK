# -*- coding: utf-8 -*-

import re

#from gimmik.solvers.euler import euler_l
from gimmik.linear import linadv_n
from gimmik.hyperns import hyper 
from gimmik.utils import safe_src


class FluxMapping(object):
    def __init__(self, func, name, macro, jac, ndims, idx, sep, **kwargs):
        self.func = func
        self.name = name
        self.jac = jac
        self.macro = "" if macro == None else macro
        self.ndims = ndims
        self.var = int(idx[0])
        self.sep = sep

        #self.conflictable = locals()
        #self.conflictable.pop('func')

        self.sub = f'{sep}'.join(i for i in idx[1:])
        self.sub = f'{sep}'+self.sub if self.sub != '' else ''

        self.mapping, self.F = eval(f'{func}(name=name, macro=self.macro, ndims=ndims, v=self.var, sub=self.sub, **kwargs)')

        #if self._map_conflict():
        #    raise ValueError('Gimmik: flux mapping conflict')

    @safe_src
    def _substitute(self, body, mapping):
        body_o = ''
        while body != body_o:
            body_o = body
            for m in mapping:
                body = re.sub(rf'\b({m})\b', f'{mapping[m]}', body)
        return body

    @safe_src
    def build_src(self):

        src = ''
        for count, i in enumerate(self.F):
            sub_str = self._substitute(body=self.F[i], mapping=self.mapping)
            src = src + f'({self.jac[i]})*{sub_str}' + ('', '+')[count < len(self.F)-1]
        self.src = src

        return src


def src(context, func, name, jac, ndims, idx, macro=None, sep=',', **kwargs):
    flux_map = FluxMapping(func=func, name=name, macro=macro, jac=jac, ndims=ndims, idx=idx, sep=sep, **kwargs)

    return flux_map.build_src()
