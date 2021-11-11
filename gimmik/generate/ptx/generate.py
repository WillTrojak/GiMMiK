# -*- coding: utf-8 -*-

import numpy as np

from gimmik.generate.ptx import type_sizes
from gimmik.generate.ptx.array import PTXArrayShared, PTXArrayValue
from gimmik.generate.ptx.constant import PTXConstant
from gimmik.generate.ptx.manager import PTXManager
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
                src += ACC[0].mul(y, self.manager.regs[X.X[i][2]])
            else:
                src += ACC[i % n_accum].fma(y, self.manager.regs[X.X[i][2]],
                                ACC[(i-1)%n_accum])
        
        src += z.address_reg(0, warp)
        src += self.manager.regs[z.X[0][1]].st_global(ACC[(i-1) % n_accum], z.type)

        return src

    def const_dotp_half(self, X, Y, z, n_accum=1, htype='f16', full_accum=True):
        src = ''
        src += X.load_array_half(htype=htype)

        ACC = []
        ACCH = []
        for i in range(n_accum):
            acc = self.manager.new_register(X.type)
            ACC.append(self.manager.regs[acc])

        src += z.address_reg(0)

        if full_accum:
            ACCH = []
            for i in range(n_accum):
                acc = self.manager.new_register(htype)
                ACCH.append(self.manager.regs[acc])

            for i, y in enumerate(Y):
                src += ACCH[i % n_accum].mul(y, self.manager.regs[X.X[i][2]])
                if i < n_accum:
                    src += ACC[i % n_accum].mov(ACCH[i % n_accum])
                else:
                    src += ACC[i % n_accum].add(ACC[i % n_accum], ACCH[i % n_accum])

            for i in range(n_accum-1):
                src += ACC[n_accum].add(ACC[n_accum], ACC[i])

            src += self.manager.regs[z.X[0][1]].st_global(ACC[n_accum], z.type)
        else:
            for i, y in enumerate(Y):
                if i == 0:
                    src += ACC[i % n_accum].mul(y, self.manager.regs[X.X[i][2]])
                else:
                    src += ACC[i % n_accum].fma(y, self.manager.regs[X.X[i][2]],
                                ACC[(i-1)%n_accum])

            accf = self.manager.new_register(X.type)
            src += accf.cvt(ACC[(i-1) % n_accum])
            src += self.manager.regs[z.X[0][1]].st_global(accf, z.type)

        return src

    def init_data_address(self):
        src = ''

        n = self.manager.misc_regs['n']
        b_a = self.manager.misc_regs['b_a']
        c_a = self.manager.misc_regs['c_a']
        ldb_a = self.manager.misc_regs['ldb_a']
        ldc_a = self.manager.misc_regs['ldc_a']

        src += n.ld_param(f'{self.name}_param_0')
        src += b_a.ld_param(f'{self.name}_param_1')
        src += ldb_a.ld_param(f'{self.name}_param_2')
        src += c_a.ld_param(f'{self.name}_param_3')
        src += ldc_a.ld_param(f'{self.name}_param_4')

        b = self.manager.misc_regs['b']
        c = self.manager.misc_regs['c']
        ib = self.manager.misc_regs['ib']
        ic = self.manager.misc_regs['ic']
        ldb = self.manager.misc_regs['ldb']
        ldc = self.manager.misc_regs['ldc']

        src += b.cvta_to(b_a, 'global')
        src += c.cvta_to(c_a, 'global')

        src += ldb.mul(ldb_a, self.bsize, config='wide', type='s32')
        src += ldc.mul(ldc_a, self.bsize, config='wide', type='s32')

        el_a = self.manager.misc_regs['el_a']
        el = self.manager.misc_regs['el']

        src += el_a.mul(el, self.bsize, config='wide', type='s32')
        src += ib.add(b, el_a)
        src += ic.add(c, el_a)

        return src

    def header(self):
        src = ''

        src += self.declare_regs()
        
        t = self.manager.misc_regs['tid_x']
        nt = self.manager.misc_regs['ntid_x']
        bl = self.manager.misc_regs['ctaid_x']

        src += bl.mov('%ctaid.x')
        src += nt.mov('%ntid.x')
        src += t.mov('%tid.x')

        el = self.manager.misc_regs['el']
        src += el.mad(nt, bl, t)

        src += self.init_data_address()

        return src

    def header_split(self, n, split, rep):
        src = ''

        src += self.declare_shared()
        src += self.declare_regs()

        t = self.manager.misc_regs['tid_x']
        nt = self.manager.misc_regs['ntid_x']
        bl = self.manager.misc_regs['ctaid_x']
        w = self.manager.misc_regs['warp_id']
        l = self.manager.misc_regs['lane_id']

        src += bl.mov('%ctaid.x')
        src += nt.mov('%ntid.x')
        src += t.mov('%tid.x')
        src += l.mov('%laneid')
        src += w.div(t, 32)

        el = self.manager.misc_regs['el']
        # el = rep*32*blockIdx.x + (threadIdx.x % 32) + 32*(threadIdx.x/(32*split));
        src += el.div(t, 32*split)
        src += el.mul(32, el)
        src += el.add(el, l)
        src += el.mad(rep*32, bl, el)

        src += self.init_data_address()

        bs_l = self.manager.misc_regs['bs_l']
        bs_a = self.manager.misc_regs['bs_a']
        # bs_l = ((threadIdx.x % 32) + 32*(threadIdx.x/64))*n + bs_a;
        src += bs_l.div(t, 32*split)
        src += bs_l.mul(bs_l, 32)
        src += bs_a.rem(t, 32)
        src += bs_l.add(bs_l, bs_a)
        src += bs_a.mov('bs')
        src += bs_l.mul(bs_l, n)
        src += bs_l.add(bs_l, bs_a)

        return src

    def declare_regs(self):
        src = '//Misc registers\n'
        for key in self.manager.misc_regs:
            rtype = self.manager.misc_regs[key].rtype
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
        self.manager.new_misc_reg('bs_a', 's32')
        self.manager.new_misc_reg('warp_id', 's32')
        self.manager.new_misc_reg('lane_id', 's32')

    def if_block(self, reg, a, b, op, jp):
        src = reg.setp(a, b, op)
        src += reg.bra(jp)
        return src

    def if_end(self, jp):
        return self.bra_tgt(jp)

    def footer(self):
        return self.ret()

    def generate_mm(self, sm, M, beta, block_dim=None):

        # Some registers and if block
        self.idx_regs()
        jp = 'RANGE'
        src = self.if_block(self.manager.misc_regs['p'],
                            self.manager.misc_regs['el'],
                            self.manager.misc_regs['n'],
                            op='ge', jp=jp)

        b = self.manager.misc_regs['b']
        ib = self.manager.misc_regs['ib']
        ldb = self.manager.misc_regs['ldb']
        c = self.manager.misc_regs['c']
        ic = self.manager.misc_regs['ic']
        ldc = self.manager.misc_regs['ldc']

        # Generate main
        for j, jx in enumerate(M):
            X_idx = []
            C = []
            for i, x in enumerate(jx):
                if x != 0:
                    C.append(x)
                    X_idx.append(i)

            if X_idx:
                X = PTXArrayValue(self.manager, f'f{self.dtype}', b, ib, ldb, X=X_idx)
                z = PTXArrayValue(self.manager, f'f{self.dtype}', c, ic, ldc, X=[j])
                Y = [PTXConstant(c, f'f{self.dtype}') for c in C]

            src += self.const_dotp(X, Y, z)

        src += self.if_end(jp=jp)

        src_p = self.func_prototype(sm, block_dim)
        src_h = self.header()
        src_f = self.footer()

        return src_p + '{\n' + src_h + src + src_f + '}\n'

    def generate_mm_split(self, sm, M, beta, block_dim, split, rep, shr_max):
        src = ''

        # Calculate how much shared is available per gang
        shr_size = int(int(shr_max/self.bsize)/int(block_dim/split))
        rows, cols = self.row_col_split(M, split, shr_size)

        # Allocate the idex registers in the manager
        self.idx_reg_split()

        b = self.manager.misc_regs['b']
        ib = self.manager.misc_regs['ib']
        ldb = self.manager.misc_regs['ldb']
        c = self.manager.misc_regs['c']
        ic = self.manager.misc_regs['ic']
        ldc = self.manager.misc_regs['ldc']

        # if (i < n)
        jp = 'RANGE'
        src += self.if_block(self.manager.misc_regs['p'],
                             self.manager.misc_regs['el'],
                             self.manager.misc_regs['n'],
                             op='ge', jp=jp)

        # Initialise the shared memory 
        self.manager.init_shared('bs', self.bsize, shr_max)
        bs_l = self.manager.misc_regs['bs_l']
        bs_a = self.manager.misc_regs['bs_a']
        S = PTXArrayShared(self.manager, f'f{self.dtype}', bs_a, bs_l, shr_size)

        # Set predicates for warps
        P = []
        for j, col in enumerate(cols):
            P.append(self.manager.regs[self.manager.new_register('pred')])
            src += P[j].setp(self.manager.misc_regs['warp_id'], j, 'ne')

        # Shared load
        curr_tgt = self.manager.new_target()
        for j, col in enumerate(cols):
            src += self.bra_tgt(curr_tgt)
            curr_tgt = self.manager.new_target()
            src += P[j].bra(curr_tgt, uni=True)

            X = PTXArrayValue(self.manager, f'f{self.dtype}', b, ib, ldb, X=col)
            src += X.load_array_to_shared(j, S)
        src += self.bra_tgt(curr_tgt)

        # Synchronise
        src += self.bar_sync(0)
        
        # Do the dot product
        curr_tgt = self.manager.new_target()
        for j, row in enumerate(rows):
            src += self.bra_tgt(curr_tgt)
            curr_tgt = self.manager.new_target()
            src += P[j].bra(curr_tgt, uni=True)
            src += f'// if(warp == {j})\n'

            for r in row:
                X_idx = []
                C = []
                for i, x in enumerate(M[r,:]):
                    if x != 0:
                        C.append(x)
                        X_idx.append(i)

                if X_idx:
                    X = PTXArrayValue(self.manager, f'f{self.dtype}', b, ib, ldb, X=X_idx)
                    z = PTXArrayValue(self.manager, f'f{self.dtype}', c, ic, ldc, X=[r])
                    Y = [PTXConstant(c, f'f{self.dtype}') for c in C]

                    src += self.const_dotp(X, Y, z, j, S=S)

        src += self.bra_tgt(curr_tgt)

        # Add 'if (i < n)' jump point and finalise
        src += self.if_end(jp)

        src_p = self.func_prototype(sm, block_dim)
        src_h = self.header_split(shr_size, split, rep)
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
