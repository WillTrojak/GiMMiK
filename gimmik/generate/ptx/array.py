# -*- coding: utf-8 -*-

from gimmik.generate.ptx import type_sizes
from gimmik.generate.ptx.provider import PTXProvider

class PTXArrayShared(PTXProvider):
    def __init__(self, manager, type, addr, addr_l) -> None:
        super().__init__()

        self.manager = manager

        self.type = type
        self.size = type_sizes[type]

        self.addr = addr
        self.addr_l = addr_l

        self.map = {}

    def load_array(self, X, warp):
        src = ''
        for i in range(len(X.X)):
            x = (X.addr, X.i, X.X[i][0], X.ld, warp)
            j = X.X[i][0]
            if x in self.manager.loaded:
                D = self.manager.loaded[x]
                X.X[i] = (j, None, D.name)
            elif X.var_id(j) in self.map:
                a = self.map(X.var_id(j))
                d = self.manager.new_register(f'f{self.size}')
                D = self.manager.regs[d]

                self.manager.add_loaded(X.addr, X.i, X.X[i][0], self.ld, D, warp)
                src += self.ld_shared(D, f'{self.addr_l} + {a}')
        return src

    def write_out(self, v, j, id):
        self.map[id] = j
        return self.st_shared(f'{self.addr_l.name} + {j}', v)


class PTXArrayValue(PTXProvider):
    def __init__(self, manager, type, addr, i, ld, X=None) -> None:
        super().__init__()

        self.manager = manager

        self.type = type
        self.size = type_sizes[type]

        self.addr = addr
        self.i = i
        self.ld = ld

        self.X = []
        for j in X:
            self.X.append((j, None, None))
        
    def load_array(self, warp=None):
        src = ''
        for i in range(len(self.X)):
            x = (self.addr, self.i, self.X[i][0], self.ld, warp)
            if x in self.manager.loaded:
                j = self.X[i][0]
                D = self.manager.loaded[x]
                self.X[i] = (j, None, D.name)
            else:
                src += self.address_reg(i)
                (j, a, d) = self.X[i]
                A = self.manager.regs[a]
                D = self.manager.regs[d]

                self.manager.add_loaded(self.addr, self.i, self.X[i][0], self.ld, D, warp)
                src += self.ld_global(D, A, config='nc.')
        return src

    def load_array_to_shared(self, warp, S: PTXArrayShared):
        src = ''
        for i in range(len(self.X)):
            x = (self.addr, self.i, self.X[i][0], self.ld, warp)
            if x in self.manager.loaded:
                j = self.X[i][0]
                D = self.manager.loaded[x]
                self.X[i] = (j, None, D.name)
            else:
                src += self.address_reg(i)
                (j, a, d) = self.X[i]
                A = self.manager.regs[a]
                D = self.manager.regs[d]

                self.manager.add_loaded(self.addr, self.i, self.X[i][0], self.ld, D, warp)
                src += self.ld_global(D, A, config='nc.')

            src += S.write_out(D, j, self.var_id(j))
        return src

    def var_id(self, j):
        return (self.addr, self.i, j, self.ld)

    def address_reg(self, i, warp=None):
        (j, a, v) = self.X[i]
        v = self.manager.new_register(f'f{self.size}')

        if warp is None:
            Adj = [(self.addr, j-1, None), (self.addr, j, None), (self.addr, j+1, None)]
        else: 
            Adj = [(self.addr, j-1, warp), (self.addr, j, warp), (self.addr, j+1, warp),
                   (self.addr, j-1, None), (self.addr, j, None), (self.addr, j+1, None)]

        for k, a in enumerate(Adj):
            if a in self.manager.addr_reg:
                A = self.manager.regs[self.manager.addr_reg[a]]

                if k == 0 or k == 3:
                    a2 = self.manager.new_register(f's64')
                    A2 = self.manager.regs[a2]
                    self.X[i] = (j, a2, v)
                    self.manager.addr_reg[(self.addr, j, warp)] = a2
                    return self.iadd(A2, A, self.ld)
                elif k == 1 or k == 4:
                    self.X[i] = (j, self.manager.addr_reg[a], v)
                    return ''
                elif k == 2 or k == 5:
                    a2 = self.manager.new_register(f's64')
                    A2 = self.manager.regs[a2]
                    self.X[i] = (j, a2, v)
                    self.manager.addr_reg[(self.addr, j, warp)] = a2
                    return self.isub(A2, A, self.ld)
        
        a = self.manager.new_register(f's64')
        self.manager.addr_reg[(self.addr, j, warp)] = a
        self.X[i] = (j, a, v)
        A = self.manager.regs[a]

        return self.imad(A, j, self.ld, self.i)
