# -*- coding: utf-8 -*-

from collections import deque
from itertools import dropwhile
import re

class LoadOptimisation(object):
    def __init__(self, block_config, var_name):
        
        self.block_config = block_config
        self.var_name = var_name

    def apply(self, source):
        source_copy = source

        pattern = rf'\s{self.var_name}\[.*\];'

        glb_occurance = dict.fromkeys(re.findall(pattern, source), 0)

        for g in re.finditer(pattern, source):
            glb_occurance[g.group(0)] += 1

        sorted_glb_occurance = sorted(glb_occurance.items(), key=lambda x: x[1], reverse=True)

        L1_acc = 0
        for (g, n) in sorted_glb_occurance:
            if n == 1:
                ld_mode = 'lu'
            else:
                ld_mode = 'g'
            #elif L1_acc < self.block_config.L1_per_thread:
            #    L1_acc += 1
            #    ld_mode = 'ca'
            #else:
            #    ld_mode = 'cg'
            addr = ((re.search(r'\[.*\]', g)).group(0))[1:-1]
            glb_occurance[g] = (n, ld_mode, addr)

            ldg = f'__ld{ld_mode}({self.var_name} + {addr});'
            source_copy = source_copy.replace(g, ldg)

        return source_copy


class StoreOptimisation(object):
    def __init__(self, var_name, st_mode='wt'):
        
        self.var_name = var_name
        self.st_mode = st_mode

    def apply(self, source):
        if self.st_mode is not None:
            source_copy = source

            pattern = rf'\s{self.var_name}\[.*\] = .*;'

            glb_occurance = dict.fromkeys(re.findall(pattern, source), 0)

            for g in glb_occurance:
                addr = ((re.search(rf'{self.var_name}\[.*\] =',g)).group(0))[len(self.var_name)+1:-3]
                val = ((re.search(rf'=.*;',g)).group(0))[1:-1]
                stg = f'\n__st{self.st_mode}({self.var_name} + {addr}, {val});'
                source_copy = source_copy.replace(g, stg)
            return source_copy
        else: 
            return source


class LineDep(object):
    def __init__(self, line, num):
        self.line = line
        self.num = num

        self.dst = None
        self.src = []

        self.dst_dep = -1
        self.src_dep = None
        
        var_pattern = r'[\w]*\[[\w\s(*+\-,)]*\]'
        line_vars = re.finditer(var_pattern, self.line)

        for (i, v) in enumerate(line_vars):
            if i==0:
                self.dst = v.group(0)
            else:
                self.src.insert(0, v.group(0))

        self.src_dep = [-1 for s in self.src]
        self.accum = True if self.dst in self.src else False

        self.compute = self._is_compute()

    def _is_compute(self):
        if self.dst is not None:
            if len(self.src) > 1:
                return True
            elif re.search(r'[+\-*]=', self.line):
                return True
            else:
                blanked_line = self.line.replace(self.dst, '')
                for src in self.src:
                    blanked_line = blanked_line.replace(src, '')
                if len(re.findall(r'[+\-*/]', blanked_line)) >= len(self.src):
                    return True
        return False

class ComputeInterleave(object):
    def __init__(self):
        pass

    def apply(self, source):
        for (i, line) in enumerate(source.splitlines()):
            line_dep = LineDep(line, i)
            if i == 0:
                source_ll = deque([line_dep])
            else:
                self.dependancy(line_dep, source_ll)

                if line_dep.src and line_dep.compute: 
                    last_dep = max(max(line_dep.src_dep), line_dep.dst_dep)
                    if last_dep != -1:
                        for update_dep in dropwhile(lambda x: x.num<=last_dep, source_ll):
                            update_dep.num += 1
                            update_dep.dst_dep += 1
                            for j in range(len(update_dep.src_dep)):
                                if update_dep.src_dep[j] != -1:
                                    update_dep.src_dep[j] += 1
                        line_dep.num = last_dep
                        source_ll.insert(last_dep+1, line_dep)
                else:
                    source_ll.append(line_dep)

        return '\n'.join(line.line for line in source_ll)

    def dependancy(self, line, source):
        for test in source:
            if test.dst in line.src:
                line.src_dep[line.src.index(test.dst)] = test.num
            if line.dst in test.src and not(test.accum and line.accum):
                line.dst_dep = test.num
            if test.dst is not None and test.dst == line.dst:
                if not(line.accum and test.accum):
                    line.dst_dep = test.num
        return


