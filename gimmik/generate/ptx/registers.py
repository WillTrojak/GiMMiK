# -*- coding: utf-8 -*-

from gimmik.generate.ptx.constant import PTXConstant
from gimmik.utils import new_line, subclass_where
from numbers import Number
from math import log2


def select_reg(rtype, name, *args, **kwargs):
    return subclass_where(PTXBaseRegister, rtype=rtype.lower())(name, *args, **kwargs)

def value(x):
    if isinstance(x, PTXConstant):
       X = x.val
    elif isinstance(x, PTXBaseRegister):
        X = x.name
    else:
        X = x
    return X


class PTXBaseRegister(object):
    rtype = None

    def __init__(self, name) -> None:
        super().__init__()
        self.name = name

    @new_line
    def ld(self, a, ss, c=None, cop=None, level=None, vec=None):
        A = value(a)
        cfg = [ss, cop, level, vec]
        config = '.'.join('{x}'.format(x=x) for x in cfg if x is not None)
        if c is not None:
            return f'ld.{config}.{self.rtype} {self.name}, [{A} + {c}]'
        else:
            return f'ld.{config}.{self.rtype} {self.name}, [{A}]'

    def ld_global(self, a, c=None, cop=None, level=None, vec=None):
        return self.ld(a, 'global', c, cop, level, vec)

    def ld_param(self, a):
        return self.ld(a, 'param', None, None, None, None)

    def ld_shared(self, a, c=None, cop=None, level=None, vec=None):
        return self.ld(a, 'shared', c, cop, level, vec)

    @new_line
    def ldu(self, a, ss):
        pass

    @new_line
    def mov(self, a, flag=None, c=None):
        A = value(a)
        if flag == 'addr':
            return f'mov.{self.rtype} {self.name}, [{A}]'
        elif flag == 'addrc':
            return f'mov.{self.rtype} {self.name}, [{A} + {c}]'
        else:
            return f'mov.{self.rtype} {self.name}, {A}'

    @new_line
    def st(self, s, ss, type, c=None, cop=None, level=None, vec=None):
        S = value(s)
        cfg = [ss, cop, level, vec]
        config = '.'.join('{x}'.format(x=x) for x in cfg if x is not None)
        if c is not None:
            return f'st.{config}.{type} [{self.name} + {c}], {S}'
        else:
            return f'st.{config}.{type} [{self.name}], {S}'

    def st_global(self, s, type, c=None, cop=None, level=None, vec=None):
        return self.st(s, 'global', type, c, cop, level, vec)

    def st_shared(self, s, type, c=None, cop=None, level=None, vec=None):
        return self.st(s, 'shared', type, c, cop, level, vec)


class PTXPredicateRegister(PTXBaseRegister):
    rtype = "pred"
    size = None
    
    def __init__(self, name) -> None:
        super().__init__(name)

    @new_line
    def bra(self, tgt, uni=False):
        if uni:
            return f'@{self.name} bra.uni {tgt}'
        else:
            return f'@{self.name} bra {tgt}'

    @new_line
    def setp(self, a, b, op):
        A = a.name
        B = value(b)
        return f'setp.{op}.{a.rtype} {self.name}, {A}, {B}'


class PTXB8Register(PTXBaseRegister):
    rtype = "b8"
    size = 8

    def __init__(self, name) -> None:
        super().__init__(name)


class PTXS8Register(PTXBaseRegister):
    rtype = "s8"
    size = 8

    def __init__(self, name) -> None:
        super().__init__(name)


class PTXU8Register(PTXBaseRegister):
    rtype = "u8"
    size = 8

    def __init__(self, name) -> None:
        super().__init__(name)


class PTXBinRegister(PTXBaseRegister):
    rtype = None

    def __init__(self, name) -> None:
        super().__init__(name)

    @new_line
    def shr(self, a, b):
        A = value(a)
        B = value(b)
        return f'shr.{self.rtype} {self.name}, {A}, {B}'


class PTXB16Register(PTXBinRegister):
    rtype = "b16"
    size = 16

    def __init__(self, name) -> None:
        super().__init__(name)


class PTXB16Register(PTXBinRegister):
    rtype = "b32"
    size = 32

    def __init__(self, name) -> None:
        super().__init__(name)


class PTXB64Register(PTXBinRegister):
    rtype = "b64"
    size = 64

    def __init__(self, name) -> None:
        super().__init__(name)
        self.size = 64


class PTXIntRegister(PTXBinRegister):
    rtype = None

    def __init__(self, name) -> None:
        super().__init__(name)

    @new_line
    def add(self, a, b):
        # self = a - b
        A = value(a)
        B = value(b)
        return f'add.{self.rtype} {self.name}, {A}, {B}'

    def div(self, a, b):
        # self = a / b
        A = value(a)
        B = value(b)

        if isinstance(B, Number) and ((B & (B-1) == 0) and B != 0):
            return self.shr(a, int(log2(B)))
        else:
            return f'div.{self.rtype} {self.name}, {A}, {B};\n'

    def mad(self, a, b, c, config='lo'):
        # self = a*b + c
        A = value(a)
        B = value(b)
        C = value(c)

        # Some optimisations
        if (A == 0 or B == 0) and C == 0:
            return self.mov(0)
        elif C == 0:
            return self.mul(a, b, config)
        elif A == 0 or B == 0:
            return self.mov(c)
        elif A == 1:
            return self.add(b, c)
        elif B == 1:
            return self.add(a, c)
        elif A == -1:
            return self.sub(c, b)
        elif B == -1:
            return self.sub(c, a)
        else:
            return f'mad.{config}.{self.rtype} {self.name}, {A}, {B}, {C};\n'

    def mul(self, a, b, config='lo', type=None):
        # self = a * b
        A = value(a)
        B = value(b)
        if A == 0 or B == 0:
            return self.mov(0)
        elif isinstance(A, Number) and isinstance(B, Number):
            C = int(A*B)
            return self.mov(C)
        elif type is not None:
            return f'mul.{config}.{type} {self.name}, {A}, {B};\n'
        else:
            return f'mul.{config}.{self.rtype} {self.name}, {A}, {B};\n'

    def rem(self, a, b):
        # self = a % b
        A = value(a)
        B = value(b)
        if isinstance(B, Number):
            if B == 1:
                return self.mov(0)
            elif B == 2:
                # There doesn't seem to be a nice way to do this in PTX
                pass
        return f'rem.{self.rtype} {self.name}, {A}, {B};\n'

    @new_line
    def sub(self, a, b):
        # self = a - b
        A = value(a)
        B = value(b)
        return f'sub.{self.rtype} {self.name}, {A}, {B}'


class PTXU16Register(PTXIntRegister):
    rtype = "u16"
    size = 16

    def __init__(self, name) -> None:
        super().__init__(name)


class PTXU32Register(PTXIntRegister):
    rtype = "u32"
    size = 32

    def __init__(self, name) -> None:
        super().__init__(name)

    @new_line
    def cvta(self, a, ss, c=None):
        A = value(a)
        if c is not None:          
            return f'cvta.{ss}.{self.rtype} {self.name}, {A} + {c}'
        else:
            return f'cvta.{ss}.{self.rtype} {self.name}, {A}'

    def cvta_to(self, a, ss, c=None):
        return self.cvta(a, 'to.'+ss, c)


class PTXU64Register(PTXIntRegister):
    rtype = "u64"
    size = 64

    def __init__(self, name) -> None:
        super().__init__(name)

    @new_line
    def cvta(self, a, ss, c=None):
        A = value(a)
        if c is not None:          
            return f'cvta.{ss}.{self.rtype} {self.name}, {A} + {c}'
        else:
            return f'cvta.{ss}.{self.rtype} {self.name}, {A}'

    def cvta_to(self, a, ss, c=None):
        return self.cvta(a, 'to.'+ss, c)


class PTXS16Register(PTXIntRegister):
    rtype = "s16"
    size = 16

    def __init__(self, name) -> None:
        super().__init__(name)


class PTXS32Register(PTXIntRegister):
    rtype = "s32"
    size = 32

    def __init__(self, name) -> None:
        super().__init__(name)


class PTXS64Register(PTXIntRegister):
    rtype = "s64"
    size = 64

    def __init__(self, name) -> None:
        super().__init__(name)


class PTXFloatRegister(PTXBaseRegister):
    rtype = None

    def __init__(self, name) -> None:
        super().__init__(name)

    @new_line
    def add(self, a, b, rnd, ftz):
        pass
    
    @new_line
    def cvt(self, a, rnd='rn'):
        r = rnd if rnd is not None else self.rnd
        assert r is not None
        return f'cvt.{r}.{self.rtype}.{a.rtype} {self.name}, {a.name}'

    @new_line
    def div(self, a, b, rnd, ftx):
        pass

    @new_line
    def fma(self, a, b, c, rnd, ftx):
        pass
    
    @new_line
    def mul(self, a, b, rnd, ftx):
        pass

    @new_line
    def sub(self, a, b, rnd, ftx):
        pass

    @new_line
    def _add(self, a, b, config):
        A = value(a)
        B = value(b)
        if not config:
            return f'add.{self.rtype} {self.name}, {A}, {B}'
        else:
            return f'add.{config}.{self.rtype} {self.name}, {A}, {B}'

    @new_line
    def _div(self, a, b, config):
        A = value(a)
        B = value(b)
        if not config and self.rtype == "f32":
            return f'div.full.{self.rtype} {self.name}, {A}, {B}'
        elif not config:
            return f'div.rz.{self.rtype} {self.name}, {A}, {B}'
        else:
            return f'div.{config}.{self.rtype} {self.name}, {A}, {B}'

    @new_line
    def _fma(self, a, b, c, config):
        A = value(a)
        B = value(b)
        C = value(c)
        if not config:
            return f'fma.rz.{self.rtype} {self.name}, {A}, {B}, {C}'
        else:
            return f'fma.{config}.{self.rtype} {self.name}, {A}, {B}, {C}'

    @new_line
    def _mul(self, a, b, config):
        A = value(a)
        B = value(b)
        if not config:
            return f'mul.{self.rtype} {self.name}, {A}, {B}'
        else:
            return f'mul.{config}.{self.rtype} {self.name}, {A}, {B}'

    @new_line
    def _sub(self, a, b, config):
        A = value(a)
        B = value(b)
        if not config:
            return f'sub.{self.rtype} {self.name}, {A}, {B}'
        else:
            return f'sub.{config}.{self.rtype} {self.name}, {A}, {B}'


class PTXFloatHalfRegister(PTXFloatRegister):
    rtype = None
    size = None

    def __init__(self, name, rnd='rn', ftz=True, **kwargs) -> None:
        super().__init__(name)
        self.rnd = rnd
        self.ftz = 'ftz' if ftz else ''

    def _config(self, rnd=None, ftz=None):
        r = rnd if rnd is not None else self.rnd
        f = 'ftz' if ftz else self.ftz
        return '.'.join('{x}'.format(x=x) for x in [r, f] if x is not None)

    def add(self, a, b, rnd=None, ftz=None):
        # self = a + b
        return self._add(a, b, self._config(rnd, ftz))
    
    def fma(self, a, b, c, rnd=None, ftz=None):
        # self = a * b + c
        return self._fma(a, b, c, self._config(rnd, ftz))

    def mul(self, a, b, rnd=None, ftz=None):
        # self = a * b
        return self._mul(a, b, self._config(rnd, ftz))

    def sub(self, a, b, rnd=None, ftz=None):
        # self = a - b
        return self._sub(a, b, self._config(rnd, ftz))


class PTXF16Register(PTXFloatHalfRegister):
    rtype = "f16"
    size = 16

    def __init__(self, name, rnd='rn', ftz=True, **kwargs) -> None:
        super().__init__(name)
        self.rnd = rnd
        self.ftz = 'ftz' if ftz else ''


class PTXBF16Register(PTXFloatHalfRegister):
    rtype = "bf16"
    size = 16

    def __init__(self, name, rnd='rn', ftz=True, **kwargs) -> None:
        super().__init__(name)
        self.rnd = rnd
        self.ftz = 'ftz' if ftz else ''


class PTXFloatHalfx2Register(PTXFloatHalfRegister):
    rtype = None
    size = None

    def __init__(self, name, rnd='rn', ftz=True, **kwargs) -> None:
        super().__init__(name)
        self.rnd = rnd
        self.ftz = 'ftz' if ftz else ''

    @new_line
    def cvt(self, a, b, rnd='rn'):
        r = rnd if rnd is not None else self.rnd
        assert r is not None
        return f'cvt.{r}.{self.rtype}.{a.rtype} {self.name}, {a.name}, {b.name}'

    @new_line
    def pack(self, a, b):
        A = value(a)
        B = value(b)
        return f'mov.b32 {self.name}, {{{A}, {B}}}'

    @new_line
    def unpack(self, a, b):
        A = value(a)
        B = value(b)
        return f'mov.b32 {{{A}, {B}}}, {self.name}'


class PTXF16x2Register(PTXFloatHalfx2Register):
    rtype = "f16x2"
    size = 32

    def __init__(self, name, rnd='rn', ftz=True, **kwargs) -> None:
        super().__init__(name)
        self.rnd = rnd
        self.ftz = 'ftz' if ftz else ''


class PTXBF16x2Register(PTXFloatHalfx2Register):
    rtype = "bf16x2"
    size = 32

    def __init__(self, name, rnd='rz', ftz=True, **kwargs) -> None:
        super().__init__(name)
        self.rnd = rnd
        self.ftz = 'ftz' if ftz else ''


class PTXF32Register(PTXFloatRegister):
    rtype = "f32"
    size = 32

    def __init__(self, name, rnd='rz', ftz=True, approx=True) -> None:
        super().__init__(name)
        self.rnd = rnd
        self.ftz = 'ftz' if ftz else ''
        self.approx = 'approx' if approx else ''

    def _config(self, rnd=None, ftz=None):
        r = rnd if rnd is not None else self.rnd
        f = 'ftz' if ftz else self.ftz
        return '.'.join('{x}'.format(x=x) for x in [r, f] if x is not None)

    def add(self, a, b, rnd=None, ftz=None):
        # self = a + b
        return self._add(a, b, self._config(rnd, ftz))

    def div(self, a, b, rnd=None, ftz=None, approx=None):
        # self = a / b
        x = approx if approx is not None else self.approx
        if x:
            r = None
        else:
            r = rnd if rnd is not None else self.rnd
        f = 'ftz' if ftz is not None else self.ftz
        config = '.'.join('{z}'.format(z=z) for z in [r, x, f] if z is not None)

        return self._div(a, b, config)
    
    def fma(self, a, b, c, rnd=None, ftz=None):
        # self = a * b + c
        return self._fma(a, b, c, self._config(rnd, ftz))

    def mul(self, a, b, rnd=None, ftz=None):
        # self = a * b
        return self._mul(a, b, self._config(rnd, ftz))

    def sub(self, a, b, rnd=None, ftz=None):
        # self = a - b
        return self._sub(a, b, self._config(rnd, ftz))


class PTXF64Register(PTXFloatRegister):
    rtype = "f64"
    size = 64

    def __init__(self, name, rnd='nz', **kwargs) -> None:
        super().__init__(name)
        self.rnd = rnd

    def add(self, a, b, rnd=None, **kwargs):
        # self = a + b
        config = rnd if rnd is not None else self.rnd
        return self._add(a, b, config)

    def div(self, a, b, rnd=None, **kwargs):
        # self = a / b
        config = rnd if rnd is not None else self.rnd
        return self._div(a, b, config)
    
    def fma(self, a, b, c, rnd=None, **kwargs):
        # self = a * b + c
        config = rnd if rnd is not None else self.rnd
        return self._fma(a, b, c, config)

    def mul(self, a, b, rnd=None, **kwargs):
        # self = a * b
        config = rnd if rnd is not None else self.rnd
        return self._mul(a, b, config)

    def sub(self, a, b, rnd=None, **kwargs):
        # self = a - b
        config = rnd if rnd is not None else self.rnd
        return self._sub(a, b, config)
