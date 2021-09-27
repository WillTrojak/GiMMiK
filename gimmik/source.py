# -*- coding: utf-8 -*-

from collections import deque
from itertools import dropwhile
#from graphlib import TopologicalSorter
import re
import sympy as sy
import time
from typing import List


class SourceLine(object):
    def __init__(self, line: str, number: int):
        self.line_original = line
        self.line = line
        self.number = number
        self.local_number: int = 0

        # Compiled Regex 
        self.re_op = re.compile(r'[+\-*/]')
        self.re_accum = re.compile(r'[+\-*/]=')
        self.re_barrier = re.compile(r'\s*?__sync\w*\(\w*\)')
        self.re_vars = re.compile(r'\w*\[[\w\s(*+\-,)]*\]')
        self.re_var_name = re.compile(r'\A\w*')
        self.re_addr = re.compile(r'\[[\w\s(*+\-,)]*\]')
        self.re_comment = re.compile(r'\A\s*?//')
        self.re_ws = re.compile(r'[\w+-=*/()\[\];]+')
        self.re_pl = re.compile(r'\Apl')

        # Classify lines
        self.comment: bool = True if self.re_comment.search(line) else False
        self.ws: bool = True if not self.re_ws.search(line) else False
        self.compute: bool = False
        self.asign: bool = not self.compute
        self.accum: bool = False
        self.barrier: bool = False
        self.pipeline: bool = True if self.re_pl.search(line) else False

        line_vars = self.re_vars.finditer(line)

        self.dst = None
        self.dst_name = None
        self.dst_addr = None
        self.dst_ptr = None
        self.src = []
        self.src_name = []
        self.src_addr = []
        self.src_ptr = []

        if not self.comment and not self.ws:
            for (i, v) in enumerate(line_vars):
                full_var = v.group(0)
                name = (self.re_var_name.findall(full_var))[0]
                addr = sy.S((self.re_addr.findall(full_var))[0][1:-1])
                pointer = f'{name} + {addr}'
                simp_var = f'{name}[{addr}]'

                self.line = self.line.replace(full_var, simp_var)

                if i==0:
                    self.dst = simp_var
                    self.dst_name = name
                    self.dst_addr = addr
                    self.dst_ptr = pointer
                else:
                    if simp_var not in self.src:
                        self.src.insert(0, simp_var)
                        self.src_name.insert(0, name)
                        self.src_addr.insert(0, addr)
                        self.src_ptr.insert(0, pointer)

            self.accum = True if self.dst in self.src else False
            self.compute = self._is_compute()
            self.asign = not self.compute
            self.barrier = self._is_barrier()

        self.dep = -1

    def _is_barrier(self) -> bool:
        if self.re_barrier.search(self.line_original):
            return True
        return False

    def _is_compute(self) -> bool:
        if self.dst is not None:
            if len(self.src) > 1:
                return True
            elif self.re_accum.search(self.line):
                return True
            else:
                blanked_line = self.line.replace(self.dst, '')
                for src in self.src:
                    blanked_line = blanked_line.replace(src, '')
                if len(self.re_op.findall(blanked_line)) >= len(self.src):
                    return True
        return False

    def name_assigment(self, name: str) -> bool:
        if (self.dst_name == name) and self.asign:
            return True
        return False

    def dependent(self, comp) -> bool:
        # Finds if second line is dependent on first line
        dep = False

        if self.barrier:
            return True
        if comp.barrier:
            return True
        if self.comment or self.ws:
            return False
        if comp.comment or comp.ws:
            return False

        d1_in_S1 = (self.dst in self.src)
        d1_in_S2 = (self.dst in comp.src)
        d2_in_S1 = (comp.dst in self.src)
        d2_in_S2 = (comp.dst in comp.src)

        d1_is_d2 = (self.dst == comp.dst)

        l1_accum = d1_in_S1
        l2_accum = d2_in_S2
        l1al2a = (d1_is_d2 and l1_accum and l2_accum)     # l1 and l2 accumulate the same thing
        l1sl2a = (d1_is_d2 and not l1_accum and l2_accum) # l1 sets and l2 accumlates the same thing
        l1al2s = (d1_is_d2 and not l2_accum and l1_accum) # l1 acummulates and l2 re-sets

        dep = dep or l1sl2a or l1al2s
        dep = dep or (d1_in_S2 and not l1sl2a and not l1al2a)
        dep = dep or (d2_in_S1 and not l1al2a)
        dep = dep or (d1_is_d2 and not l1_accum and not l2_accum)

        return dep


class SourceBlock(object):
    def __init__(self, shr=False, line=None, lines=None) -> None:
        self.block_addrs = {}

        if line is not None:
            line.local_number = 0
            self.lines = deque([line])
            self._add_to_block_addrs(line)
        elif lines is not None:
            for (i, line) in enumerate(lines):
                line.local_number = i
                self._add_to_block_addrs(line)
            self.lines = lines
        else: 
            self.lines = deque([])

        self.is_shr = shr
        self.can_dep = True

    def clear_block(self) -> None:
        self.lines.clear()
        self.block_addrs.clear()
        self.shr = False
        return

    def split(self, n=-1):

        if n == -1 or n == len(self):
            return SourceBlock(shr=self.is_shr)
        
        new_block = SourceBlock(shr=self.is_shr)

        ln = len(self)
        for i in range(n + 1, ln):
            new_block.append(self.lines[n+1])
            self.remove(self.lines[n+1], reform=False)

        self._reform_block_addrs()

        return new_block

    def split_lines(self, line_idx):
        new_block = SourceBlock(shr=self.is_shr)

        for (n, i) in enumerate(iter(set(line_idx))):
            line = self.lines[i - n]
            self.remove(line, reform=False)
            new_block.append(line)

        self._reform_block_addrs()

        return new_block

    def remove(self, line, reform=True):
        self.lines.remove(line)
        for i in range(line.local_number,len(self)):
            self.lines[i].local_number = i
        if reform: # useful if popping multiple
            self._reform_block_addrs(v)
        return 

    def append(self, line: SourceLine) -> None:
        line.local_number = len(self.lines)
        self.lines.append(line)
        self._add_to_block_addrs(line)
        return

    def appendleft(self, line: SourceLine) -> None:
        for addr in self.block_addrs:
            self.block_addrs[addr] = [x + 1 for x in self.block_addrs[addr]]
        self.lines.appendleft(line)
        for (i, l) in enumerate(self.lines):
            l.local_number = i
        self._add_to_block_addrs(line)
        return

    def append_block(self, block) -> None:
        for line in block.lines:
            self.append(line)
        return

    def rearange_lines(self, dep: List[int], lines: SourceLine) -> None:
        new_lines = deque([])
        for (i, j) in enumerate(dep):
            new_lines.append(lines[j])
            new_lines[i].local_number = i
        self.lines = new_lines
        self._reform_block_addrs()
        return

    def _reform_block_addrs(self) -> None:
        self.block_addrs.clear()
        for line in self.lines:
            self._add_to_block_addrs(line)
        return

    def _add_to_block_addrs(self, line: SourceLine) -> None:
        if line.dst is not None:
            line_nums = self.block_addrs.get(line.dst, [])
            line_nums.append(line.local_number)
            self.block_addrs[line.dst] = line_nums
        for src in line.src:
            line_nums = self.block_addrs.get(src, [])
            line_nums.append(line.local_number)
            self.block_addrs[src] = line_nums
        return

    # def get_dep_graph(self):
    #     if self.can_dep:
    #         self.dep_graph = TopologicalSorter()
    #         for (i, line) in enumerate(self.lines):
    #             for j in range(i):
    #                 test = self.lines[j]
    #                 if test.dependent(line):
    #                     self.dep_graph.add(line.local_number, test.local_number)
    #         return [*self.dep_graph.static_order()]
    #     else:
    #         return [line.local_number for line in self.lines]

    def __len__(self):
        return len(self.lines)