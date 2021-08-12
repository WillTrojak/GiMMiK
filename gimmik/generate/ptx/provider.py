# -*- coding: utf-8 -*-

from gimmik.generate.ptx.memory import PTXConstant, PTXRegister
from numbers import Number
from math import log2

class PTXProvider(object):
    def __init__(self) -> None:
        super().__init__()

        self.fma_mod = {'f32': 'rn.ftz', 'f64': 'rn'}

    def _value(self, x):
        if isinstance(x, PTXConstant):
            X = x.valh
        elif isinstance(x, PTXRegister):
            X = x.name
        else:
            X = x
        return X

    def bra(self, p, tgt):
        if p.type != 'pred':
            raise TypeError('GiMMiK PTX: branch reg not predicate')
        return f'@{p.name} bra {tgt};\n'

    def bra_tgt(self, tgt):
        return f'{tgt}:\n'

    def bar_sync(self, b, num=None):
        if b >= 16:
            raise ValueError('GiMMIK PTX: barrier num out of range')
        num_threads = f', {num}' if num is not None else ''
        return f'bar.sync {b}{num_threads};\n'

    def convert_to(self, space, ri, ro):
        if ri.type != ro.type:
            raise TypeError('In and out types do not match')
        return f'cvta.to.{space}.{ri.type} {ro.name}, {ri.name};\n'

    def convert_to_global(self, ri, ro):
        return self.convert_to('global', ri, ro)

    def convert_to_shared(self, ri, ro):
        return self.convert_to('shared', ri, ro)
        
    def convert_to_local(self, ri, ro):
        return self.convert_to('local', ri, ro)

    def convert_to_const(self, ri, ro):
        return self.convert_to('const', ri, ro)

    def fma(self, z, a, b, c):
        # z = a*b + c
        A = self._value(a)
        B = self._value(b)
        C = self._value(c)
        return f'fma.{self.fma_mod[z.type]}.{z.type} {z.name}, {A}, {B}, {C};\n'

    def iadd(self, d, a, b):
        # d = a + b
        A = self._value(a)
        B = self._value(b)
        return f'add.{d.type} {d.name}, {A}, {B};\n'

    def idiv(self, d, a, b):
        A = self._value(a)
        B = self._value(b)
        if isinstance(B, Number) and ((B & (B-1) == 0) and B != 0):
            return self.shr(d, a, int(log2(B)))
        else:
            return f'div.{d.type} {d.name}, {A}, {B};\n'

    def imul(self, d, a, b, config='lo'):
        # d = a*b
        A = self._value(a)
        B = self._value(b)
        return f'mul.{config}.{d.type} {d.name}, {A}, {B};\n'

    def imad(self, d, a, b, c, config='lo'):
        # d = a*b + c
        A = self._value(a)
        B = self._value(b)
        C = self._value(c)

        # Some optimisations
        if (A == 0 or B == 0) and C == 0:
            return self.mov(d, 0)
        elif C == 0:
            return self.imul(d, a, b, config)
        elif A == 0 or B == 0:
            return self.mov(d, c)
        elif A == 1:
            return self.iadd(d, b, c)
        elif B == 1:
            return self.iadd(d, a, c)
        elif A == -1:
            return self.isub(d, c, b)
        elif B == -1:
            return self.isub(d, c, a)
        else:
            return f'mad.{config}.{d.type} {d.name}, {A}, {B}, {C};\n'

    def isub(self, d, a, b):
        # d = a - b
        A = self._value(a)
        B = self._value(b)
        return f'sub.{d.type} {d.name}, {A}, {B};\n'

    def load(self, ltype, d, a, config=''):
        A = self._value(a)
        return f'ld.{ltype}.{config}{d.type} {d.name}, [{A}];\n'

    def ld_global(self, d, a, config=''):
        return self.load('global', d, a, config)

    def ld_shared(self, d, a, config=''):
        return self.load('shared', d, a, config)

    def mov(self, d, v):
        V = self._value(v)
        return f'mov.{d.type} {d.name}, {V};\n'

    def mul(self, d, a, b):
        A = self._value(a)
        B = self._value(b)
        return f'mul.{self.fma_mod[d.type]}.{d.type} {d.name}, {A}, {B};\n'

    def selp(self, d, a, b, p):
        A = self._value(a)
        B = self._value(b)
        return f'selp.{d.type} {d.name}, {A}, {B}, {p.name};\n'

    def setp(self, p, a, b, op):
        A = self._value(a)
        B = self._value(b)
        return f'setp.{op}.{a.type} {p.name}, {A}, {B};\n'

    def shr(self, d, a, b):
        A = self._value(a)
        B = self._value(b)
        return f'shr.{d.type} {d.name}, {A}, {B};\n'

    def store(self, stype, a, s, config=''):
        A = self._value(a)
        S = self._value(s)
        return f'st.{stype}.{config}{s.type} [{A}], {S};\n'

    def st_global(self, a, s, config=''):
        return self.store('global', a, s, config)

    def st_shared(self, a, s, config=''):
        return self.store('shared', a, s, config)
