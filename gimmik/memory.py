# -*- coding: utf-8 -*-

#from gimmik.utils import new_line, safe_src

import numpy as np

def safe_src(func):
    def wrapper(*args, **kwargs):
        return f'({func(*args, **kwargs)})'
    return wrapper

def new_line(func):
    def wrapper(*args, **kwargs):
        return f'{func(*args, **kwargs)}\n'
    return wrapper

def is_contiguous(l):
    v0 = l[0]
    for (i, v) in enumerate(l):
        contig = (v0 + i) == v
        if not contig:
            return False
    return True


class BaseMemory(object):
    def __init__(self):
        pass

    def is_local(self):
        return False

    def is_shared(self):
        return False

    def is_global(self):
        return False

    def point(self):
        pass


class GlobalMemory(BaseMemory):
    def __init__(self, name, elem, p, nvars, thrd_v, ld, macro_name):
        super(GlobalMemory, self).__init__()
        self.name = name
        
        # Number of point in a line _not_ order
        self.p = p
        self.elem = elem
        self.nvars = nvars
        self.thrd_v = thrd_v
        self.ld = ld
        self.macro = macro_name

    def is_global(self):
        return True

    def point(self, v, x_const, thread_dim=None, name=None, ld=None):
        if name is None:
            name = self.name
        if ld is None:
            ld = self.ld

        if thread_dim is None:
            source = f'{name}[{self.macro}({self.elem}, {v}) + ({x_const})*{ld}]'
        else:
            x = f'{x_const} + {self.p**thread_dim}*{self.thrd_v}'
            source = f'{name}[{self.macro}({self.elem}, {v}) + ({x})*{ld}]'
        return source

    def dst(self, v, x_const, thread_dim=None, name=None, ld=None):
        return self.point(v, x_const, thread_dim, name, ld)

    @new_line
    def global_write(self, src, dst):
        return f'{dst} = {src};'

class RegisterMemory(object):
    def __init__(self, name, size, rid):
        self.name = name
        self.size = size
        self.rid = rid

        self.addr_queue = [] # filo queue of stores (x, v, addr)
        self.x2addr = {} # dictionary point to addr. { (x, v): addr }
        self.addr2x = {} # dictionary addr to point. { addr: (x, v) }
        self.free_addr = [i for i in range(size)] # list of free addresses

        # Bool used to track use in accumulation
        self.first = [True for i in range(size)]

    def point(self, i):
        return f'{self.name}[{i}]'

    def reset_accumulator(self, i='all'):
        if i == 'all':
            for j in range(self.size): 
                self.first[j] = True
        else:
            self.first[i] = True
        return

class LocalMemory(BaseMemory):
    def __init__(self, name, p, nvars, size):
        super(LocalMemory, self).__init__()
        self.name = name
        self.p = p
        self.nvars = nvars
        self.size = size

        self.thread_reg = [RegisterMemory(name, size, thread) for thread in range(self.p)]

    def clear_space(self, n, priority=False):
        for thread in range(self.p):
            for i in range(n):
                (j, v, addr) = self.thread_reg[thread].addr_queue.pop(0)
                self.thread_reg[thread].free_addr.append(addr)
                self.thread_reg[thread].x2addr.pop((j, v))
                self.thread_reg[thread].addr2x.pop(addr)
        
        return

    @new_line
    def copy_to_local(self, src, dst):
        return f'{dst} = {src};'

    def free_reg(self, v, x_const, thread_dim):
        for thread in range(self.p):
            x = x_const + (self.p**thread_dim)*thread
            addr = self.thread_reg[thread].x2addr.get((x, v))
            self.thread_reg[thread].free_addr.append(addr)
        return

    def is_local(self):
        return True

    def is_stored(self, v, x_const, thread_dim=None):
        if thread_dim is None:
            stored = True
            for thread in range(self.p):
                x = (x_const, v)
                stored = stored and (self.thread_reg[thread].x2addr.get(x, False) is not False)
        else:
            stored = True
            for thread in range(self.p):
                x = (x_const +  (self.p**thread_dim)*thread, v)
                stored = stored and (self.thread_reg[thread].x2addr.get(x, False) is not False)
            return stored

    def is_full(self):
        full = False
        for thread in range(self.p):
            full = full or (len(self.thread_reg[thread].free_addr) <= 0)
        return full

    def is_space(self, n):
        space = True
        for thread in range(self.p):
            space = space and (len(self.thread_reg[thread].free_addr) >= n)
        return space

    def new_local_dst(self, v, x_const, thread_dim):

        if not self.is_space(n=1):
            self.clear_space(n=1)

        # Log memory addres and usage
        for thread in range(self.p):
            x = x_const + (self.p**thread_dim)*thread
            
            addr = self.thread_reg[thread].free_addr.pop(0)

            # If register already in use clear dictionary entry
            x_tuple = self.thread_reg[thread].addr2x.get(addr, False)
            if x_tuple is not False:
                self.thread_reg[thread].x2addr.pop(x_tuple)
                self.thread_reg[thread].addr2x.pop(addr)

            # Log new memory usage
            self.thread_reg[thread].x2addr[(x, v)] = addr
            self.thread_reg[thread].addr2x[addr] = (x, v)

            self.thread_reg[thread].addr_queue.append((x, v, addr))

        return f'{self.thread_reg[0].name}[{addr}]'

    def point(self, v, x_const, thread_dim=None):
        addr = self.thread_reg[0].x2addr.get((x_const, v))
        return f'{self.name}[{addr}]'


class CacheStack(object):
    def __init__(self):
        self.addr = {}   # Addr -> x
        self.x = {}      # x    -> addr
        self.stack = []  # stack of addr

    def push(self, addr, x):
        self.addr[addr] = x
        self.x[x] = addr
        self.stack.insert(0, addr)
        return

    def del_x(self, x):
        if self.x.get(x, False) is not False:
            addr = self.x[x]
            self.x.pop(x)
            self.addr.pop(addr)
            try:
                self.stack.remove(addr)
            except ValueError:
                pass
            return addr
        return -1

    def del_addr(self, addr):
        if self.addr.get(addr, False) is not False:
            x = self.addr[addr]
            self.x.pop(x)
            self.addr.pop(addr)
            try:
                self.stack.remove(addr)
            except ValueError:
                pass
            return x
        return -1

    def pop(self, i=0):
        addr = self.stack.pop(i)
        x = self.del_addr(addr)
        return addr, x


class SharedMemory(BaseMemory):
    def __init__(self, name, p, nvars, offset, thrd_v, size):
        super(SharedMemory, self).__init__()
        self.name = name

        # Number of point in a line _not_ order
        self.p = p  
        self.nvars = nvars
        self.offset = offset
        self.thrd_v = thrd_v
        self.size = size # max number of members


        # Syncronisation Tracker
        self.syncron = np.zeros((self.p**3, self.nvars, self.p), dtype=np.int8)

        self.nlow = 2
        self.nmed = 1
        self.nhigh = 0

        self.high = CacheStack()
        self.med = CacheStack()
        self.low = CacheStack()
        self.free = [i for i in range(self.size)]
        self.addr = {} # (x, v) -> addr

    def contig_addr(self, n, priority=None):
        priority = self.nlow if priority is None else priority

        if len(self.free) >= n:
            self.free.sort()
            for (i, a) in enumerate(self.free):
                if is_contiguous(self.free[i:i+n]):
                    return [self.free.pop(i) for j in range(n)]
        elif (priority <= self.nmed) and (len(self.low.stack) >= n):
            caddr = []
            for j in range(n):
                addr, x = self.low.pop(0)
                self.addr.pop(x)
                caddr.insert(j, addr)
            caddr.sort()
            return caddr
        elif (priority <= self.nhigh) and (len(self.med.stack) >= n):
            caddr = []
            for j in range(n):
                addr, x = self.med.pop(0)
                self.addr.pop(x)
                caddr.insert(j, addr)
            caddr.sort()
            return caddr

        raise ValueError('GiMMiK: no contiguous memory available')

    @new_line
    def copy_to_share(self, v, x_const, thread_dim, src, dst):
        # log that shared is now not synced
        for thread in range(self.p):
            x = x_const + (self.p**thread_dim)*thread
            self.syncron[x,v,thread] += 1
        return f'{dst} = {src};'

    def set_priority(self, priority, v, x_const, thread_dim=None):
        if thread_dim is None:
            x = x_const
            (addr, priority_o) = self.addr.get((x, v))

            old_stack = self.stack_select(priority_o)
            new_stack = self.stack_select(priority)
            if new_stack != old_stack:
                xo = old_stack.del_addr(addr)
                new_stack.push(addr, xo)
                self.addr[(x, v)] = (addr, priority)

        else:
            for thread in range(self.p):
                x = x_const + (self.p**thread_dim)*thread
                (addr, priority_o) = self.addr.get((x, v))

                old_stack = self.stack_select(priority_o)
                new_stack = self.stack_select(priority)
                if new_stack != old_stack:
                    xo = old_stack.del_addr(addr)
                    new_stack.push(addr, xo)
                    self.addr[(x, v)] = (addr, priority)
        return

    def is_shared(self):
        return True

    def is_space(self, n, priority=None):
        priority = self.nlow if priority is None else priority

        if priority == self.nlow:
            space = (len(self.free) >= n)
        elif priority == self.nmed:
            space = (len(self.free) >= n) or (len(self.low.stack) >= n)
        elif priority == self.nhigh:
            space = ((len(self.free) >= n) or 
                     (len(self.low.stack) >= n) or
                     (len(self.med.stack) >= n))
        return space

    def is_stored(self, v, x_const, thread_dim=None):
        if thread_dim is None:
            stored = self.addr.get((x_const, v), False) is not False
        else:
            stored = True
            for thread in range(self.p):
                x = x_const + (self.p**thread_dim)*thread
                stored = (self.addr.get((x, v), False) is not False) and stored
        return stored

    def is_synced(self, v, x_const, thread_dim=None):
        if thread_dim is None:
            synced = True
            for thread in range(self.p):
                synced = synced and (self.syncron[x_const,v,thread] == self.syncron[x_const,v,0])
            return synced
        else:
            synced = True
            for thread in range(self.p):
                for i in range(self.p):
                    x = x_const + pow(self.p, thread_dim)*i
                    synced = synced and (self.syncron[x,v,thread] == self.syncron[x,v,0])
            return synced

    def new_shared_dst(self, v, x_const, thread_dim, priority=None):
        priority = self.nlow if priority is None else priority

        addr = self.contig_addr(n=self.p, priority=priority)
        stack = self.stack_select(priority)

        # Log memory addres and usage
        for thread in range(self.p):
            x = x_const + (self.p**thread_dim)*thread
            
            self.addr[(x, v)] = (addr[thread], priority)
            stack.push(addr=addr[thread], x=(x, v))

        return self.point(v, x_const, thread_dim)

    def point(self, v, x_const, thread_dim=None):
        (addr, priority) = self.addr.get((x_const, v))
        if thread_dim is None:
            return f'{self.name}[{addr} + {self.offset}]'
        else:
            return f'{self.name}[{addr} + {self.thrd_v} + {self.offset}]'

    def stack_select(self, priority):
        if priority == self.nlow:
            stack = self.low
        elif priority == self.nmed:
            stack = self.med
        elif priority == self.nhigh:
            stack = self.high
        return stack
