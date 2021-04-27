# -*- coding: utf-8 -*-

class BaseManager(object):
    def __init__(self):
        
        self.state_count = 0
        
        self.n_local = 0
        self.local = []

        self.n_share = 0
        self.share = []
        self.glb = [] # glb[0] is input memory, glb[1] is output memory.

    def set_shr_priority(self, priority, v, x_const, thread_dim=None):
        in_share = [shr.is_stored(v, x_const, thread_dim) for shr in self.share]
        if any(in_share):
            for (i, val) in enumerate(in_share):
                if val == True:
                    shr = self.share[i]
                    shr.set_priority(priority, v, x_const, thread_dim)
        return

    def location(self, v, x_const, thread_dim):
        # Get the name of the best current location of a variable
        # If in global this _won't_ attempt to write to heirachy,
        # global read is intend on really as a fall back and the main
        # use of this is creating flux function mappings

        mem = self.memory_unit(v, x_const, thread_dim)

        return mem.point(v, x_const, thread_dim)

    def memory_unit(self, v, x_const, thread_dim):
        in_share = [shr.is_stored(v, x_const, thread_dim) for shr in self.share]
        in_local = [lcl.is_stored(v, x_const, thread_dim) for lcl in self.local]

        if any(in_local):
            mem = self.local[next(i for (i,v) in enumerate(in_local) if v == True)]
        elif any(in_share):
            mem = self.share[next(i for (i,v) in enumerate(in_share) if v == True)]
        else:
            mem = self.glb[0]

        return mem

    def next_usable_shared(self, n, priority):
        shr_usable = [shr.is_space(n, priority=priority) for shr in self.share]
        
        if any(shr_usable):
            next_usable = next(i for (i,v) in enumerate(shr_usable) if v == True)
        else:
            next_usable = None

        return next_usable

    def read_to_local(self, lcl, priority, v, x_const, thread_dim=None, shr_bypass=False):
        source = ''
        
        in_shared = [shr.is_stored(v, x_const, thread_dim) for shr in self.share]
        mem = self.memory_unit(v, x_const, thread_dim)

        if mem.is_local():
            if lcl.is_stored(v, x_const, thread_dim):   
                dst = lcl.point(v, x_const, thread_dim)
            else:
                dst = mem.point(v, x_const)
                #dst = lcl.new_local_dst(v, x_const, thread_dim)
                #source += lcl.copy_to_local(src, dst)

            # Oportunistic Shared Store
            if not any(in_shared) and not shr_bypass:
                source += self.read_to_shared(priority, v, x_const, thread_dim, dst)
        elif mem.is_shared() or mem.is_global():
            src = mem.point(v, x_const, thread_dim)
            dst = lcl.new_local_dst(v, x_const, thread_dim)
            source += lcl.copy_to_local(src, dst)

            # Oportunistic Shared Store
            if not any(in_shared) and not shr_bypass:
                source += self.read_to_shared(priority, v, x_const, thread_dim, dst)

        return source

    def read_to_shared(self, priority, v, x_const, thread_dim=None, src=None):

        shr_usable = [shr.is_space(n=shr.p, priority=priority) for shr in self.share]
        in_shared = [shr.is_stored(v, x_const, thread_dim) for shr in self.share]
        in_local = [lcl.is_stored(v, x_const, thread_dim) for lcl in self.local]

        if src is None:
            if any(in_shared):
                return ''
            elif any(in_local):
                lcl = self.local[next(i for (i,v) in enumerate(in_local) if v == True)]
                src = lcl.point(v, x_const)
            else:
                src = self.glb[0].point(v, x_const, thread_dim)
            
        if not any(shr_usable):
            source = ''
        else:
            shr = self.share[next(i for (i,v) in enumerate(shr_usable) if v == True)]
            dst = shr.new_shared_dst(v, x_const, thread_dim, priority)
            source = shr.copy_to_share(v, x_const, thread_dim, src, dst)

        return source

    def variable_map(self, x_const, thread_dim=None):
        mapping = {}
        for v in range(self.nvars):
            mapping[v] = self.location(v, x_const, thread_dim)
        return mapping

    # For debugging
    def _write_state_heading(self, file, h, level=0):
        file.write(f'{h}\n')
        if level == 0:
            div = '='
        elif level == 1:
            div = '*'
        elif level == 2:
            div = '~'
        else:
            div = '-'

        file.write(f'{div * 80}\n')
    
        return

    # For debugging
    def write_state(self, note=None):
        filename = f'mem_state_{self.state_count}'
        f = open(filename+'.txt', 'w')
        
        self._write_state_heading(f, f'GiMMik Memory Manager, Memory State: {self.state_count}', level=0)
        f.write('(x, v, address)\n')
        if note is not None:
            f.write(f'Note: {note}\n')
        f.write('\n')


        for (j, shr) in enumerate(self.share):

            self._write_state_heading(f, f'Shared Memory: {j}', level=0)
            f.write(f'Shared Max Size = {shr.size}\n')
            nallocated = len(shr.high.addr) + len(shr.med.addr) + len(shr.low.addr)
            f.write(f'Total Allocated: {nallocated} ({nallocated*100/shr.size}%)\n')

            f.write('\n')
            self._write_state_heading(f, 'Shared Memory, High Priority Partition', level=1)
            f.write(f'Allocated: {len(shr.high.addr)} ({len(shr.high.addr)*100/shr.size}%)\n')
            for (i, a) in enumerate(shr.high.addr):
                f.write(f'{a}: {shr.high.addr[a]}\n')
            f.write('\n')

            f.write('\n')
            self._write_state_heading(f, 'Shared Memory, Med Priority Partition', level=1)
            f.write(f'Allocated: {len(shr.med.addr)} ({len(shr.med.addr)*100/shr.size}%)\n')
            for (i, a) in enumerate(shr.med.addr):
                f.write(f'{a}: {shr.med.addr[a]}\n')
            f.write('\n')

            f.write('\n')
            self._write_state_heading(f, 'Shared Memory, Low Priority Partition', level=1)
            f.write(f'Allocated: {len(shr.low.addr)} ({len(shr.low.addr)*100/shr.size}%)\n')
            for (i, a) in enumerate(shr.low.addr):
                f.write(f'{a}: {shr.low.addr[a]}\n')
            f.write('\n')

        self._write_state_heading(f, 'Register Memory', level=0)
        for (j, lcl) in enumerate(self.local):
            f.write('\n')

            self._write_state_heading(f, f'Register Memory: {j}', level=1)
            f.write(f'Register Max Size = {lcl.size}\n\n')
            for thread in range(lcl.p):
                self._write_state_heading(f, f'Register {j}: Thread {thread}', level=2)
                allocated = len(lcl.thread_reg[thread].addr_queue)
            
                f.write(f'Allocated: {allocated} ({allocated*100/lcl.thread_reg[thread].size}%)\n')
                for (i, a) in enumerate(lcl.thread_reg[thread].addr_queue):
                    f.write(f'{i}: {a}\n')
                f.write('\n')

        f.close()

        self.state_count += 1

        return f'// Memory state in {filename}\n'