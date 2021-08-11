# -*- coding: utf-8 -*-

import numpy as np

from gimmik.generate.ptx import type_sizes
from gimmik.generate.ptx.array import PTXArrayShared, PTXArrayValue
from gimmik.generate.ptx.manager import PTXManager
from gimmik.generate.ptx.memory import PTXConstant
from gimmik.generate.ptx.provider import PTXProvider


class GimmikPTXFunction(PTXProvider):
    def __init__(self, name, precision) -> None:
        super().__init__()

        self.name = name
        self.precision = precision
        self.dtype = 32 if precision == np.float32 else 64
        self.bsize = int(self.dtype/8)

        self.manager = PTXManager('x')

    def const_dotp(self, X, Y, z, warp=None, n_accum=1, S=None):
        src = ''
        if S is not None:
            src += S.load_array(X, warp)
        src += X.load_array(warp)

        ACC = []
        for i in range(n_accum):
            acc = self.manager.new_register(X.type)
            ACC.append(self.manager.regs[acc])

        for i, y in enumerate(Y):
            if i == 0:
                src += self.mul(ACC[0], y, self.manager.regs[X.X[i][2]])
            else:
                src += self.fma(ACC[i % n_accum], y, self.manager.regs[X.X[i][2]],
                                ACC[(i-1)%n_accum])
        
        src += z.address_reg(0)
        src += self.st_global(self.manager.regs[z.X[0][1]], ACC[0])

        return src

    def header(self, sm):
        src = ''
        src += f'''
.version 7.2
.target sm_{sm}
.address_size 64

.visible .entry {self.name}(
	.param .u32 {self.name}_param_0,
	.param .u64 {self.name}_param_1,
	.param .u32 {self.name}_param_2,
	.param .u64 {self.name}_param_3,
	.param .u32 {self.name}_param_4
)\n {{\n'''

        src += '//Misc registers\n'
        for key in self.manager.misc_regs:
            rtype = self.manager.misc_regs[key].type
            rname = self.manager.misc_regs[key].name
            src += f'.reg .{rtype} {rname};\n'

        src += '//Main registers\n'
        for key in type_sizes:
            rnum = self.manager._type_id[key]
            if rnum > 0:
                src += f'.reg .{key} {self.manager.mem_name}_{key}_<{rnum+1}>;\n'

        src += f'''
mov.u32	ctaid_x, %ctaid.x;
mov.u32	ntid_x, %ntid.x;
mov.u32	tid_x, %tid.x;
mad.lo.s32 el, ntid_x, ctaid_x, tid_x;
ld.param.u32 n, [{self.name}_param_0];

ld.param.u64 b_a, [{self.name}_param_1];
cvta.to.global.u64 b, b_a;
ld.param.u64 c_a, [{self.name}_param_3];
cvta.to.global.u64 c, c_a;
mul.wide.s32 el_a, el, {self.bsize};
add.s64 ib, b, el_a;
add.s64 ic, c, el_a;

ld.param.s32 ldb_a, [{self.name}_param_2];
mul.wide.s32 ldb, ldb_a, {self.bsize};
ld.param.s32 ldc_a, [{self.name}_param_4];
mul.wide.s32 ldc, ldc_a, {self.bsize};
'''

        return src

    def header_split(self, sm, block_dim=None):
        src = ''
        src += f'''
.version 7.2
.target sm_{sm}
.address_size 64

.visible .entry {self.name}(
	.param .u32 {self.name}_param_0,
	.param .u64 {self.name}_param_1,
	.param .u32 {self.name}_param_2,
	.param .u64 {self.name}_param_3,
	.param .u32 {self.name}_param_4
)\n'''

        if block_dim is not None:
            src += f'.maxntid {block_dim}, 1, 1\n'
        src += '{\n'

        src += '//Misc registers\n'
        for key in self.manager.misc_regs:
            rtype = self.manager.misc_regs[key].type
            rname = self.manager.misc_regs[key].name
            src += f'.reg .{rtype} {rname};\n'

        src += '//Main registers\n'
        for key in type_sizes:
            rnum = self.manager._type_id[key]
            if rnum > 0:
                src += f'.reg .{key} {self.manager.mem_name}_{key}_<{rnum+1}>;\n'

        src += f'''
mov.u32	ctaid_x, %ctaid.x;
mov.u32	ntid_x, %ntid.x;
mov.u32	tid_x, %tid.x;
mad.lo.s32 el, ntid_x, ctaid_x, tid_x;
shr.s32 warp_id, tid_x, 5;
ld.param.u32 n, [{self.name}_param_0];

ld.param.u64 b_a, [{self.name}_param_1];
cvta.to.global.u64 b, b_a;
ld.param.u64 c_a, [{self.name}_param_3];
cvta.to.global.u64 c, c_a;
mul.wide.s32 el_a, el, {self.bsize};
add.s64 ib, b, el_a;
add.s64 ic, c, el_a;

ld.param.s32 ldb_a, [{self.name}_param_2];
mul.wide.s32 ldb, ldb_a, {self.bsize};
ld.param.s32 ldc_a, [{self.name}_param_4];
mul.wide.s32 ldc, ldc_a, {self.bsize};
'''

        return src

    def idx_regs(self):
        self.manager.new_misc_reg('p', 'pred')
        
        self.manager.new_misc_reg('n', 'u32')
        self.manager.new_misc_reg('el', 's32')
        self.manager.new_misc_reg('el_a', 's64')

        self.manager.new_misc_reg('b_a', 'u64')
        self.manager.new_misc_reg('b', 'u64')
        self.manager.new_misc_reg('c_a', 'u64')
        self.manager.new_misc_reg('c', 'u64')

        self.manager.new_misc_reg('ib', 's64')
        self.manager.new_misc_reg('ic', 's64')
        self.manager.new_misc_reg('ldb_a', 's32')
        self.manager.new_misc_reg('ldc_a', 's32')
        self.manager.new_misc_reg('ldb', 's64')
        self.manager.new_misc_reg('ldc', 's64')

        self.manager.new_misc_reg('ctaid_x', 'u32')
        self.manager.new_misc_reg('ntid_x', 'u32')
        self.manager.new_misc_reg('tid_x', 'u32')

    def idx_reg_split(self):
        self.manager.new_misc_reg('p', 'pred')
        
        self.manager.new_misc_reg('n', 'u32')
        self.manager.new_misc_reg('el', 's32')
        self.manager.new_misc_reg('el_a', 's64')

        self.manager.new_misc_reg('b_a', 'u64')
        self.manager.new_misc_reg('b', 'u64')
        self.manager.new_misc_reg('c_a', 'u64')
        self.manager.new_misc_reg('c', 'u64')

        self.manager.new_misc_reg('ib', 's64')
        self.manager.new_misc_reg('ic', 's64')
        self.manager.new_misc_reg('ldb_a', 's32')
        self.manager.new_misc_reg('ldc_a', 's32')
        self.manager.new_misc_reg('ldb', 's64')
        self.manager.new_misc_reg('ldc', 's64')

        self.manager.new_misc_reg('ctaid_x', 'u32')
        self.manager.new_misc_reg('ntid_x', 'u32')
        self.manager.new_misc_reg('tid_x', 'u32')
        self.manager.new_misc_reg('warp_id', 's32')


    def if_block(self, reg, a, b, op, jp):
        src = f'setp.{op}.s32 {reg.name}, {a.name}, {b.name};\n'
        src += self.bra(reg, jp)
        return src

    def if_end(self, jp):
        return self.bra_tgt(jp)

    def footer(self):
        src = 'ret;\n'
        src += '}\n'
        return src

    def generate_mm(self, sm, M, beta):

        # Some registers and if block
        self.idx_regs()
        jp = 'BLOCK0'
        src = self.if_block(self.manager.misc_regs['p'],
                            self.manager.misc_regs['el'],
                            self.manager.misc_regs['n'],
                            op='ge', jp=jp)

        # Generate main
        for j, jx in enumerate(M):
            X_idx = []
            C = []
            for i, x in enumerate(jx):
                if x != 0:
                    C.append(x)
                    X_idx.append(i)

            if X_idx:
                X = PTXArrayValue(self.manager, f'f{self.dtype}', 'b', 'ib', 'ldb', X=X_idx)
                z = PTXArrayValue(self.manager, f'f{self.dtype}', 'b', 'ic', 'ldc', X=[j])
                Y = [PTXConstant(c, self.dtype) for c in C]

            src += self.const_dotp(X, Y, z)

        src += self.if_end(jp=jp)

        src_h = self.header(sm)
        src_f = self.footer()

        return src_h + src + src_f

    def generate_mm_split(self, sm, M, beta, block_dim, split, rep):
        rows = self.row_split(M)
        cols = self.col_split_shared(M, block_dim, split)

        self.idx_reg_split()

        S = PTXArrayShared(self.manager, f'f{self.dtype}', 'bs', 'bs_l')

        # Shared load
        for j, col in enumerate(cols):
            X = PTXArrayValue(self.manager, 'f32', 'b', 'ib', 'ldb', X=col)

            X.load_array_to_shared(j, S)            

def generator(context, sm, M, beta, name, dtype, block_dim, split=None, rep=None):
    func = GimmikPTXFunction(name, dtype)

    if split is None:
        src = func.generate_mm(sm, M, beta)
    else:
        src = func.generate_mm_split(sm, M, beta, block_dim, split, rep)

    return src
