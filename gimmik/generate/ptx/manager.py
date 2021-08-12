# -*- coding: utf-8 -*-

from gimmik.generate.ptx.memory import PTXRegister


class PTXManager(object):
    def __init__(self, mem_name='x', tgt_base='TGT') -> None:
        super().__init__()

        self.mem_name = mem_name
        self.tgt_base = tgt_base

        self.tgt_num = 0

        self.regs = {}
        self.misc_regs = {}
        self._type_id = {'s8': 0, 's16': 0, 's32': 0, 's64': 0,
                         'u8': 0, 'u16': 0, 'u32': 0, 'u64': 0,
                         'f32': 0, 'f64': 0,
                         'b8': 0, 'b16': 0, 'b32': 0, 'b64': 0,
                         'pred': 0,
                        }

        self.loaded = {}
        self.addr_reg = {}
        self.addr_reg_shr = {}
        
        self.shr_name = None
        self.shr_align = None
        self.shr_addr_type = None
        self.shr_static = None

    def add_loaded(self, addr, col, row, ld, reg, warp=0):
        x = (addr, col, row, ld, warp)
        self.loaded[x] = reg

    def new_register(self, type):
        id = self._type_id[type] = self._type_id[type] + 1
        name = self.reg_name(type, id)
        
        self.regs[name] = PTXRegister(name, type)

        return name

    def init_shared(self, name, align, size, atype='b8', static=True):
        self.shr_name = name
        self.shr_align = align
        self.shr_addr_type = atype
        self.shr_static = static
        self.shr_max = size
        if not static:
            raise ValueError('GiMMIK PTX: Only static shared memory supported')

    def new_target(self):
        id = self.tgt_num = self.tgt_num + 1
        return self.tgt_name(id)

    def new_misc_reg(self, name, type):
        self.misc_regs[name] = PTXRegister(name, type)

    def reg_name(self, type, number):
        return f'{self.mem_name}_{type}_{number}'

    def tgt_name(self, number):
        return f'{self.tgt_base}_{number}'

    @property
    def reg_file_usage(self):
        used = 0
        for v in self.regs:
            used += self.regs[v].size
        return used

    def num_reg(self, rtype):
        i = 0
        for key in self.regs:
            reg = self.regs[key]
            if reg.type == rtype:
                i += 1
        return i
