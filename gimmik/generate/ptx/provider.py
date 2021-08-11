# -*- coding: utf-8 -*-

from gimmik.generate.ptx.memory import PTXConstant, PTXRegister


class PTXProvider(object):
    def __init__(self) -> None:
        super().__init__()

        self.fma_mod = {'f32': 'rn.ftz', 'f64': 'rn'}

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
        A = a.valh if isinstance(a, PTXConstant) else a.name
        B = b.valh if isinstance(b, PTXConstant) else b.name
        C = c.valh if isinstance(c, PTXConstant) else c.name
        return f'fma.{self.fma_mod[z.type]}.{z.type} {z.name}, {A}, {B}, {C};\n'

    def iadd(self, d, a, b):
        # d = a + b
        A = a.name if isinstance(a, PTXRegister) else a
        B = b.name if isinstance(b, PTXRegister) else b
        return f'add.{d.type} {d.name}, {A}, {B};\n'

    def imul(self, d, a, b, config='lo'):
        # d = a*b
        A = a.name if isinstance(a, PTXRegister) else a
        B = b.name if isinstance(b, PTXRegister) else b
        return f'mul.{config}.{d.type} {d.name}, {A}, {B};\n'

    def imad(self, d, a, b, c, config='lo'):
        # d = a*b + c
        A = a.name if isinstance(a, PTXRegister) else a
        B = b.name if isinstance(b, PTXRegister) else b
        C = c.name if isinstance(c, PTXRegister) else c
        return f'mad.{config}.{d.type} {d.name}, {A}, {B}, {C};\n'

    def isub(self, d, a, b):
        # d = a - b
        A = a.name if isinstance(a, PTXRegister) else a
        B = b.name if isinstance(b, PTXRegister) else b
        return f'sub.{d.type} {d.name}, {A}, {B};\n'

    def load(self, ltype, d, a, config=''):
        A = a.name if isinstance(a, PTXRegister) else a
        return f'ld.{ltype}.{config}{d.type} {d.name}, [{A}];\n'

    def ld_global(self, d, a, config=''):
        return self.load('global', d, a, config)

    def ld_shared(self, d, a, config=''):
        return self.load('shared', d, a, config)

    def mov(self, d, v):
        if isinstance(v, PTXConstant):
            return f'mov.{d.type} {d.name}, {v.valh};\n'
        elif isinstance(v, PTXRegister):
            return f'mov.{d.type} {d.name}, {v.name};\n'

    def mul(self, d, a, b):
        A = a.valh if isinstance(a, PTXConstant) else a.name
        B = b.valh if isinstance(b, PTXConstant) else b.name
        return f'mul.{self.fma_mod[d.type]}.{d.type} {d.name}, {A}, {B};\n'

    def store(self, stype, a, s, config=''):
        S = s.valh if isinstance(s, PTXConstant) else s.name
        A = a.name if isinstance(a, PTXRegister) else a
        return f'st.{stype}.{config}{s.type} [{A}], {S};\n'

    def st_global(self, a, s, config=''):
        return self.store('global', a, s, config)

    def st_shared(self, a, s, config=''):
        return self.store('shared', a, s, config)
