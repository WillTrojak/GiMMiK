# -*- coding: utf-8 -*-

import struct


class PTXConstant(object):
    def __init__(self, value, type) -> None:
        super().__init__()

        self.true_value = value

        int_types = ['b16', 'b32', 'b64',
                     's16', 's32', 's64',
                     'u16', 'u32', 'u64',
                    ]
        float_types = ['f32', 'f64']

        assert type in int_types + float_types

        if type == 'f32':
            self.val = self._fp32_to_hexstr(value)
        elif type == 'f64':
            self.val = self._fp64_to_hexstr(value)
        elif type in int_types:
            self.val = value

    def _fp64_to_hexstr(self, f):
        if f == 0:
            return '0d00000000'
        else:
            h = hex(struct.unpack('>Q', struct.pack('>d', f))[0])
            return str(h).replace('x', 'd')

    def _fp32_to_hexstr(self, f):
        if f == 0:
            return '0f0000'
        else:
            h = hex(struct.unpack('>I', struct.pack('>f', f))[0])
            return str(h).replace('x', 'f')
