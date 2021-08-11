# -*- coding: utf-8 -*-

import struct

from gimmik.generate.ptx import type_sizes

class PTXConstant(object):
    def __init__(self, value, size=64) -> None:
        super().__init__()

        self.valt = value
        if size == 64:
            if value == 0:
                self.valh = '0d00000000'
            else:
                self.valh = self.fp64_to_hexstr(value)
        elif size == 32:
            if value == 0:
                self.valh = '0f0000'
            else:
                self.valh = self.fp32_to_hexstr(value)
        else:
            raise ValueError('Unsupported constant size')

    def fp64_to_hexstr(self, f):
        h = hex(struct.unpack('>Q', struct.pack('>d', f))[0])
        return str(h).replace('x', 'd')

    def fp32_to_hexstr(self, f):
        h = hex(struct.unpack('>I', struct.pack('>f', f))[0])
        return str(h).replace('x', 'f')


class PTXRegister(object):
    def __init__(self, name, type) -> None:
        super().__init__()

        self.name = name
        self.type = type
        self.size = type_sizes[type]