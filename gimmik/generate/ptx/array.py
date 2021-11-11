# -*- coding: utf-8 -*-

from gimmik.generate.ptx import type_sizes
from gimmik.generate.ptx.provider import PTXProvider

class PTXArrayShared(PTXProvider):
    def __init__(self, manager, type, addr, addr_l, max_size) -> None:
        super().__init__()

        self.manager = manager

        self.type = type
        self.size = type_sizes[type]
        self.bsize = int(self.size/8)

        self.addr = addr
        self.addr_l = addr_l

        self.map = {}
        self.num = 0
        self.max_size = max_size

    def load_array(self, X, warp):
        src = ''

        for i in range(len(X.X)):
            x = (X.addr, X.i, X.X[i][0], X.ld, warp)
            j = X.X[i][0]
            if x in self.manager.loaded:
                D = self.manager.loaded[x]
                X.X[i] = (j, None, D.name)
            elif X.var_id(j) in self.map:
                a = self.map[X.var_id(j)]
                d = self.manager.new_register(f'f{self.size}')
                D = self.manager.regs[d]

                self.manager.add_loaded(X.addr, X.i, X.X[i][0], X.ld, D, warp)
                src += D.ld_shared(self.addr_l, c=a)
        return src

    def new_shared(self):
        num = self.num
        self.num += 1
        return num

    def write_out(self, v, j, id):
        self.map[id] = j*self.bsize
        return self.addr_l.st_shared(v, self.type, c=j*self.bsize)

    def write_out_new(self, v, id):
        j = self.new_shared()
        self.map[id] = j*self.bsize
        return self.addr_l.st_shared(v, self.type, c=j*self.bsize)

class PTXArrayValue(PTXProvider):
    def __init__(self, manager, type, addr, i, ld, X=None, temp=None) -> None:
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

        self.temp = temp
        
    def load_array(self, warp=None):
        src = ''
        for i in range(len(self.X)):
            x = (self.addr, self.i, self.X[i][0], self.ld, warp)
            if x in self.manager.loaded:
                j = self.X[i][0]
                D = self.manager.loaded[x]
                self.X[i] = (j, None, D.name)
            else:
                src += self.address_reg(i, warp)
                (j, a, d) = self.X[i]
                A = self.manager.regs[a]
                D = self.manager.regs[d]

                self.manager.add_loaded(self.addr, self.i, self.X[i][0], self.ld, D, warp)
                src += D.ld_global(A, cop='nc')
        return src

    def load_array_half(self, warp=None, htype='f16'):
        src = ''
        for i in range(len(self.X)):
            x = (self.addr, self.i, self.X[i][0], self.ld, warp)
            if x in self.manager.loaded:
                j = self.X[i][0]
                D = self.manager.loaded[x]
                self.X[i] = (j, None, D.name)
            else:
                src += self.address_reg(i, warp, rtype=htype)
                (j, a, d) = self.X[i]
                A = self.manager.regs[a]
                D = self.manager.regs[d]

                self.manager.add_loaded(self.addr, self.i, self.X[i][0], self.ld, D, warp)
                src += self.temp.ld_global(A, cop='nc')
                src += D.cvt(self.temp)
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
                src += self.address_reg(i, warp)
                (j, a, d) = self.X[i]
                A = self.manager.regs[a]
                D = self.manager.regs[d]

                self.manager.add_loaded(self.addr, self.i, self.X[i][0], self.ld, D, warp)
                src += D.ld_global(A, cop='nc')

            src += S.write_out_new(D, self.var_id(j))
        return src

    def var_id(self, j):
        return (self.addr, self.i, j, self.ld)

    def address_reg(self, i, warp=None, rtype=None):
        (j, a, v) = self.X[i]
        rtype = f'f{self.size}' if rtype is None else rtype
        v = self.manager.new_register(rtype)

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
                    return A2.add(A, self.ld)
                elif k == 1 or k == 4:
                    self.X[i] = (j, self.manager.addr_reg[a], v)
                    return ''
                elif k == 2 or k == 5:
                    a2 = self.manager.new_register(f's64')
                    A2 = self.manager.regs[a2]
                    self.X[i] = (j, a2, v)
                    self.manager.addr_reg[(self.addr, j, warp)] = a2
                    return A2.sub(A, self.ld)
        
        a = self.manager.new_register(f's64')
        self.manager.addr_reg[(self.addr, j, warp)] = a
        self.X[i] = (j, a, v)
        A = self.manager.regs[a]

        return A.mad(j, self.ld, self.i)
