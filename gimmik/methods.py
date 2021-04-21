# -*- coding: utf-8 -*-

from gimmik.fluxcls import BaseFlux
from gimmik.euler import EulerFlux
from gimmik.hyperns import HyperNSFlux
from gimmik.linear import LinearFlux

from gimmik.manager import BaseManager
from gimmik.matrix import GimmikMatrix
from gimmik.memory import GlobalMemory, LocalMemory, SharedMemory
from gimmik.utils import new_line, safe_src

def select_flux(func, ndims, fargs):
    if func == 'linadv':
        return LinearFlux(ndims, fargs)
    elif func == 'euler':
        return EulerFlux(ndims, fargs)
    elif func == 'hyperns':
        return HyperNSFlux(ndims, fargs)
    else:
        raise ValueError(f'Gimmik: Unknown flux type requested - {func}')


class Planar3dMemoryManger(BaseManager):
    def __init__(self, glb_mem, glb_out_mem, shr_mem, lcly_mem, lclx_mem, acc_reg):
        super().__init__()

        self.nvars = glb_mem.nvars
        self.glb.insert(0, glb_mem)
        self.glb.insert(1, glb_out_mem)
        self.share.insert(0, shr_mem)
        #self.share.insert(1, shrd_mem)
        self.local.insert(0, lclx_mem)
        self.local.insert(1, lcly_mem)
        
        self.acc_reg = acc_reg

    @new_line
    def accumulate(self, reg, i, src, op='+'):
        operator = f'= {op}' if reg.first[i] else f'{op}='
        reg.first[i] = False
        return f'{reg.point(i)} {operator} ({src});'


    #def read_to_hierachy(self, v, x_const, thread_dim=None, priority=False, shr_bypass=False):
    #    shared_full = not self.shr_mem.is_space(self.shr_mem.p)
    #    yreg_full = self.lcly_mem.is_full()
    #    xreg_full = self.lclx_mem.is_full()


class Plane3d(object):
    def __init__(self, A, p, nvars, ndims, thrd_v, mem, flux_func, fargs):
        self.src = ''

        self.A = A
        
        # Number of point in a line _not_ order
        self.p = p
        
        self.nvars = nvars

        # Name of variable that stores thread in element group number
        self.thrd_v = thrd_v 

        # memory manager
        self.mem = mem

        self.flux = select_flux(flux_func, ndims, fargs)

    def build(self, mem_debug=False):
        source = self.src

        for yz_plane in range(self.p):
            source += self.mem.write_state(note=f'Top of plane loop, before read, yz_plane={yz_plane}')
            source += self._read_y_line(yz_plane)
            source += self._read_yz_plane(yz_plane)
            source += self.mem.write_state(note=f'Top of plane loop, after read, yz_plane={yz_plane}')

            # y-line loop
            for y_line in range(self.p):
                self.mem.acc_reg.reset_accumulator()
                # x-point loop
                jac = ['1.','ze','ze']
                for x in range(self.p):
                    i = x + self.p*y_line
                    priority = 1 if x == yz_plane + 1 else 0
                    source += self._read_x_point(x, y_line, priority=priority)
                    
                    q = self.mem.variable_map(i, thread_dim=2)
                    
                    for v in range(self.nvars):
                        flux = self.flux.build_flux(q, v, jac)
                        mat = self.A.matrix_value(yz_plane, x)
                        acc = f'{mat}*{flux}'
                        source += self.mem.accumulate(self.mem.acc_reg, v, acc)

                source += self.mem.write_state(note=f'After x loop, y_line={y_line}, yz_plane={yz_plane}')


                source += self._add_warp_sync()

                # y and z contribution
                for v in range(self.nvars):
                    acc_yz = ''

                    for i in range(self.p):
                        # y-line
                        idx = yz_plane + self.p*i
                        q = self.mem.variable_map(idx, thread_dim=2)
                        jac = ['ze','1.','ze']
                        flux = self.flux.build_flux(q, v, jac)
                        mat = self.A.matrix_value(i, y_line)
                        acc_yz += f'+({mat}*{flux})'

                        # z-line
                        idx = yz_plane + self.p*y_line + self.p*self.p*i
                        q = self.mem.variable_map(idx)
                        jac = ['ze','ze','1.']
                        flux = self.flux.build_flux(q, v, jac)
                        mat = self.A.matrix_value(i, self.thrd_v)
                        acc_yz += f'+({mat}*{flux})'

                    source += self.mem.accumulate(self.mem.acc_reg, v, acc_yz[1:])

                    source += self.mem.glb[0].global_write(self.mem.acc_reg.point(v), 
                        self.mem.glb[1].point(v, x_const=yz_plane+y_line*self.p,
                             thread_dim=2))

            if yz_plane != self.p:
                self._yz_plane_priority(yz_plane, yz_plane+1)
        self.src = source
        return source

    @new_line
    def _add_warp_sync(self):
        return '__syncwarp();'

    @new_line
    def _read_x_point(self, x, y, priority):
        source = ''
        for v in range(self.nvars):
            i = x + y*self.p
            source += self.mem.read_to_local(lcl=self.mem.local[0], priority=priority, 
                v=v, x_const=i, thread_dim=2, shr_bypass=False)
        return source

    @new_line
    def _read_y_line(self, x):
        source = ''

        for y in range(self.p):
            i = x + self.p*y
            for v in range(self.nvars):
                source += self.mem.read_to_local(self.mem.local[1], priority=0, v=v, 
                    x_const=i, thread_dim=2, shr_bypass=False)
        return source

    @new_line
    def _read_yz_plane(self, x):
        source = ''

        # threads read along y, for thead on dfferent z
        for y in range(self.p):
            i = x + self.p*y
            for v in range(self.nvars):
                source += self.mem.read_to_shared(priority=0, v=v, x_const=i, thread_dim=2)

        return source

    def _yz_plane_priority(self, x_old, x_new):
        for y in range(self.p):
            i_old = x_old + self.p*y
            i_new = x_new + self.p*y
            for v in range(self.nvars):
                self.mem.set_shr_priority(priority=2, v=v, x_const=i_old, thread_dim=2)
                self.mem.set_shr_priority(priority=0, v=v, x_const=i_new, thread_dim=2)
        return