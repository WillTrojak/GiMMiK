# -*- coding: utf-8 -*-

from gimmik.utils import ncols
import numpy as np
from math import ceil

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

    def header(self):
        src = ''

        src += self.declare_regs()

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

    def header_split(self, n, split):
        src = ''

        src += self.declare_shared()
        src += self.declare_regs()

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

        d = self.manager.misc_regs['bs_l']
        t = self.manager.misc_regs['tid_x']
        src += self.idiv(d, t, 32*split)
        src += self.imul(d, d, n)

        return src

    def declare_regs(self):
        src = '//Misc registers\n'
        for key in self.manager.misc_regs:
            rtype = self.manager.misc_regs[key].type
            rname = self.manager.misc_regs[key].name
            src += f'.reg .{rtype} {rname};\n'

        src += '//Main registers\n'
        for key in type_sizes:
            rnum = self.manager._type_id[key]
            if rnum > 0:
                src += f'.reg .{key} {self.manager.mem_name}_{key}_<{rnum+1}>;\n'
        return src

    def declare_shared(self):
        src = ''
        m = self.manager
        if m.shr_name is not None:
            if m.shr_static:
                size = m.shr_max - (m.shr_max % m.shr_align)
                src += f'.shared .align {m.shr_align} .{m.shr_addr_type} {m.shr_name}[{size}];\n'
            else:
                src += f'.extern .align {m.shr_align} .{m.shr_addr_type} {m.shr_name}[];\n'
        return src

    def func_prototype(self, sm, block_dim=None, version=7.2):
        src = f'''
.version {version}
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
        self.idx_regs()
        
        self.manager.new_misc_reg('bs_l', 's32')
        self.manager.new_misc_reg('warp_id', 's32')

    def if_block(self, reg, a, b, op, jp):
        src = f'setp.{op}.s32 {reg.name}, {a.name}, {b.name};\n'
        src += self.bra(reg, jp)
        return src

    def if_end(self, jp):
        return self.bra_tgt(jp)

    def footer(self):
        return 'ret;\n'

    def generate_mm(self, sm, M, beta, block_dim=None):

        # Some registers and if block
        self.idx_regs()
        jp = 'RANGE'
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
                z = PTXArrayValue(self.manager, f'f{self.dtype}', 'c', 'ic', 'ldc', X=[j])
                Y = [PTXConstant(c, self.dtype) for c in C]

            src += self.const_dotp(X, Y, z)

        src += self.if_end(jp=jp)

        src_p = self.func_prototype(sm, block_dim)
        src_h = self.header(sm)
        src_f = self.footer()

        return src_p + '{\n' + src_h + src + src_f + '}\n'

    def generate_mm_split(self, sm, M, beta, block_dim, split, rep, shr_max):
        src = ''

        # Calculate how much shared is available per gang
        shr_size = int(int(shr_max/self.bsize)/int(block_dim/split))
        rows, cols = self.row_col_split(M, split, shr_size)

        # Allocate the idex registers in the manager
        self.idx_reg_split()

        # if (i < n)
        jp = 'RANGE'
        src += self.if_block(self.manager.misc_regs['p'],
                             self.manager.misc_regs['el'],
                             self.manager.misc_regs['n'],
                             op='ge', jp=jp)

        # Initialise the shared memory 
        self.manager.init_shared('bs', self.bsize, shr_max)
        S = PTXArrayShared(self.manager, f'f{self.dtype}', 'bs', 'bs_l', shr_size)

        # Set predicates for warps
        P = []
        for j, col in enumerate(cols):
            P.append(self.manager.regs[self.manager.new_register('pred')])
            src += self.setp(P[j], self.manager.misc_regs['warp_id'], j, 'ne')

        # Shared load
        curr_tgt = self.manager.new_target()
        for j, col in enumerate(cols):
            src += self.bra_tgt(curr_tgt)
            curr_tgt = self.manager.new_target()
            src += self.bra(P[j], curr_tgt)

            X = PTXArrayValue(self.manager, 'f32', 'b', 'ib', 'ldb', X=col)
            src += X.load_array_to_shared(j, S)
        src += self.bra_tgt(curr_tgt)

        # Synchronise
        src += self.bar_sync(0)
        
        # Do the dot product
        curr_tgt = self.manager.new_target()
        for j, row in enumerate(rows):
            src += self.bra_tgt(curr_tgt)
            curr_tgt = self.manager.new_target()
            src += self.bra(P[j], curr_tgt)

            for r in row:
                X_idx = []
                C = []
                for i, x in enumerate(M[r,:]):
                    if x != 0:
                        C.append(x)
                        X_idx.append(i)

                if X_idx:
                    X = PTXArrayValue(self.manager, f'f{self.dtype}', 'b', 'ib', 'ldb', X=X_idx)
                    z = PTXArrayValue(self.manager, f'f{self.dtype}', 'c', 'ic', 'ldc', X=[r])
                    Y = [PTXConstant(c, self.dtype) for c in C]

                    src += self.const_dotp(X, Y, z, j, S=S)

        src += self.bra_tgt(curr_tgt)


        # Add 'if (i < n)' jump point and finalise
        src += self.if_end(jp)

        src_p = self.func_prototype(sm, block_dim)
        src_h = self.header_split(shr_size, split)
        src_f = self.footer()

        return src_p + '{\n' + src_h + src + src_f + '}\n'

    def row_col_split(self, M, split, n):
        cols = []
        rows = []

        c_per_s = int(np.shape(M)[1]/split)
        r_per_s = int(np.shape(M)[0]/split)

        for i in range(split):
            rows.append([x for x in range(r_per_s*i, r_per_s*(i+1))])
        rows[-1] += [i for i in range(r_per_s*split, np.shape(M)[0])]

        print(n, np.shape(M)[1])

        if n < np.shape(M)[1]:
            for i in range(split):
                cols.append([])
            nz = np.count_nonzero(M, axis=0)
            map = np.flip(np.argsort(nz))

            for i, ind in enumerate(map):
                if i < n:
                    cols[i%split] += [ind]
                else:
                    break
        else: 
            for i in range(split):
                cols.append([x for x in range(c_per_s*i, c_per_s*(i+1))])
            cols[-1] += [i for i in range(c_per_s*split, np.shape(M)[1])]

        return rows, cols


    def row_split(self, M):
        return []

def generator(context, sm, M, beta, name, dtype, block_dim, split=None,
              rep=None, shr_max=None):
    func = GimmikPTXFunction(name, dtype)

    if split is None:
        src = func.generate_mm(sm, M, beta)
    else:
        src = func.generate_mm_split(sm, M, beta, block_dim, split, rep,
                                     shr_max)

    return src
