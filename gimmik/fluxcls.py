# -*- coding: utf-8 -*-

import re

from gimmik.utils import safe_src

class BaseFlux(object):
    def __init__(self, ndims, fargs):
        self.fargs = fargs
        self.ndims = ndims

    @safe_src
    def build_flux(self, q, v, jac):
        mapping, F = self._flux(q, v)

        src = ''
        for count, i in enumerate(F):
            sub_str = self._substitute(body=F[i], mapping=mapping)
            src = src + f'({jac[i]})*{sub_str}' + ('', '+')[count < len(F)-1]
        self.src = src

        return src

    def _flux(self, q):
        pass

    @safe_src
    def _substitute(self, body, mapping):
        body_o = ''
        while body != body_o:
            body_o = body
            for m in mapping:
                body = re.sub(rf'\b({m})\b', f'{mapping[m]}', body)
        return body