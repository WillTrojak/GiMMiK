# -*- coding: utf-8 -*-

from gimmik.source import SourceBlock, SourceLine

from collections import deque
from itertools import dropwhile
#from graphlib import TopologicalSorter
import re
import sympy as sy
from typing import List

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


# class Pipeline(object):
#     def __init__(self, source, dtype, max_coag=1, compute_size=16) -> None:
#         self.dtype = dtype
#         self.dbytes = 4 if dtype == 'float' else 8

#         for (i, line) in enumerate(source.splitlines()):
#             source_line = SourceLine(line, i)
#             #print(source_line.number, source_line.dst_dep, source_line.src_dep, source_line.line)
#             if i == 0:
#                 self.src_lines = deque([source_line])
#             else:
#                 self.src_lines.append(source_line)

#         for line in self.src_lines:
#             self._dependency(line, self.src_lines)

#         self.dep_graph = self.get_dep_graph()
#         self.max_coag = max_coag
#         self.compute_size = compute_size

#     def get_dep_graph(self) -> List[int]:
#         ts = TopologicalSorter()

#         for (i, line) in enumerate(self.src_lines):
#             for j in range(i):
#                 test = self.src_lines[j]
#                 if test.dependent(line):
#                     ts.add(line.number, test.number)

#         self.ts = ts
#         return [*ts.static_order()]

#     def get_dst_dep_graph(self, dst) -> List[int]:
#         ts = TopologicalSorter()

#         for (i, line) in enumerate(self.src_lines):
#             if line.dst_name == dst:
#                 for j in range(i):
#                     test = self.src_lines[j]
#                     if test.dependent(line):
#                         ts.add(line.number, test.number)
#             else:
#                 for j in range(i):
#                     test = self.src_lines[j]
#                     ts.add(line.number, test.number)
        
#         return [*ts.static_order()]

#     def rearange_lines(self, dep, lines):
#         new_lines = deque([])
#         for (i, j) in enumerate(dep):
#             new_lines.append(lines[j])
#             new_lines[i].number = i
#         return new_lines


#     def _dependency(self, line, src_lines, local=False):
#         if line.comment or line.barrier or line.ws:
#             return
        
#         i = src_lines.index(line)
#         if not local:
#             for j in reversed(range(i-1)):
#                 test = src_lines[j]
#                 if test.dependent(line):
#                     line.dep = test.number
#                     return
#         elif local:
#             for j in reversed(range(i-1)):
#                 test = src_lines[j]
#                 if test.dependent(line):
#                     line.local_dep = test.local_number
#                     return
#         return

#     def print_lines(self):
#         return '\n'.join(line.line for line in self.src_lines)

#     def pipeline_opt(self, shr_name, pipeline_contol=True, cg_namespace='cg'):

#         self._move_shares_early(shr_name)

#         #self.shr_dep = self.get_dst_dep_graph(shr_name)
#         #self.src_lines = self.rearange_lines(self.shr_dep, self.src_lines)
                
#         # Seperate code into blocks based on share loads
#         self._block_source(shr_name)

#         self._coagulate_blocks()
        
#         self._rearrange_compute()

#         stagger = not pipeline_contol
#         self._partition_compute_blocks(shr_name, stagger)
#         self._clear_empty_blocks()

#         if pipeline_contol:
#             self._add_pipeline_control()        
#             self._comment_sync_lines()

#         return self._print_blocks()

#     def _clear_empty_blocks(self):
#         i = 0
#         while i < len(self.blocks):
#             block = self.blocks[i]
#             if len(block) == 0:
#                 self.blocks.remove(block)
#             else:
#                 i += 1
#         return

#     def _comment_sync_lines(self):
#         for block in self.blocks:
#             for line in block.lines:
#                 if line.barrier:
#                     src = '//' + line.line
#                     line.line = src
#         return

#     def _rearrange_compute(self):
#         for (i, block) in enumerate(self.blocks):
#             if len(block) > 1 and not block.is_shr:
#                 block.dep = block.get_dep_graph()
#                 block.rearange_lines(block.dep, block.lines)

#     def _move_shares_early(self, shr_name):
#         # Move share loads as early as possible.
#         for i in range(len(self.src_lines)):
#             line = self.src_lines[i]

#             if line.dst_name == shr_name:
#                 self.src_lines.remove(line)
#                 old_line_number = line.number
#                 line.number = line.dep + 1
#                 self.src_lines.insert(line.dep + 1, line)

#                 for int_line in dropwhile(lambda x: x.number<=line.dep+1, self.src_lines):
#                     if int_line.number < old_line_number:
#                         int_line.number += 1
#                     self._dependency(int_line, self.src_lines)
#         return

#     def _block_source(self, shr_name):
#         current_type = False
#         self.blocks = deque([SourceBlock()])

#         for (i, line) in enumerate(self.src_lines):
#             if line.comment or line.ws:
#                 shr_assign = current_type
#             else:
#                 shr_assign = line.name_assigment(shr_name)

#             if shr_assign and current_type:
#                 (self.blocks[-1]).append(line)
#             elif shr_assign and not current_type:
#                 current_type = True
#                 self.blocks.append(SourceBlock(shr=True, line=line))

#             if not shr_assign and not current_type:
#                 (self.blocks[-1]).append(line)
#             if not shr_assign and current_type:
#                 current_type = False
#                 self.blocks.append(SourceBlock(shr=False, line=line))

#         return

#     def _coagulate_blocks(self):
#         # Coagulate small share blocks by pushing back
#         for i in range(len(self.blocks)):
#             block = self.blocks[i]

#             if block.is_shr and (len(block.lines) <= self.max_coag):
#                 dep = False
#                 for sline in block.lines:
#                     for cline in self.blocks[i+1].lines:
#                         dep = dep or sline.dependent(cline)
#                         if dep:
#                             break
#                     if dep:
#                         break
#                 if not dep:
#                     self.blocks[i+2].append_block(block)
#                     block.clear_block()

#         # Coagulate small compute blocks by pushing forwards
#         for i in reversed(range(len(self.blocks))):
#             block = self.blocks[i]

#             if i>1 and not block.is_shr and (len(block) <= self.max_coag):
#                 dep = False
#                 for cline in block.lines:
#                     for sline in self.blocks[i-1].lines:
#                         dep = dep or cline.dependent(sline)
#                         if dep:
#                             break
#                     if dep:
#                         break
#                 if not dep:
#                     self.blocks[i-2].append_block(block)
#                     block.clear_block()

#         # Post coagulation clean-up
#         self._clear_empty_blocks()

#         i = 0
#         while i < len(self.blocks) - 1:
#             block1 = self.blocks[i]
#             block2 = self.blocks[i+1]
#             if not(block1.is_shr ^ block2.is_shr) :
#                 block1.append_block(block2)
#                 self.blocks.remove(block2)
#             else:
#                 i += 1

#         return

#     def _partition_compute_blocks(self, shr_name, stagger):

#         i = 0
#         prev_shr = None
#         while True:

#             block = self.blocks[i]

#             if not block.is_shr and prev_shr is not None:

#                 num_comp_shr = 0
#                 shr_addrs = []
#                 shr_line = []

#                 if len(prev_shr) < 1:
#                     pass
#                 else:
#                     # Get Shared addresses from line
#                     for l in range(len(block)):
#                         line = block.lines[l]

#                         new_shr_addr = []
#                         indices = [j for j, x in enumerate(line.src_name) if x == shr_name]
#                         shr_line += [prev_shr.block_addrs[line.src[j]] for j in iter(indices) 
#                                      if prev_shr.block_addrs.get(line.src[j], False)]

#                         new_shr_addr = [line.src[j] for j in indices if line.src[j] not in shr_addrs and prev_shr.block_addrs.get(line.src[j], False)]
#                         shr_addrs += new_shr_addr

#                         if len(shr_addrs) >= self.compute_size:
#                             shr_line = [z for s in shr_line for z in s]

#                             split_shr = prev_shr.split_lines(shr_line)
#                             split_comp = block.split(l)
#                             self.blocks.insert(prev_shr_idx, split_shr)
#                             prev_shr_idx += 1
#                             self.blocks.insert(i + 2, split_comp)
#                             i += 1

#                             if stagger:
#                                 old_comp = self.blocks[i-1]
#                                 self.blocks.insert(prev_shr_idx-1, old_comp)
#                                 prev_shr_idx +=1

#                             break

#                 i += 1
#             elif block.is_shr:
#                 prev_shr = block
#                 prev_shr_idx = i

#                 i += 1
#             else:
#                 i += 1

#             if i >= len(self.blocks):
#                 break

#         return

#     def _add_pipeline_control(self):
#         for (i, block) in enumerate(self.blocks):
#             prev_count = None
#             if i == 0:
#                 prev_shr = block.is_shr
#                 count = 1
#                 maxcount = 1
#             else: 
#                 if block.is_shr == prev_shr:
#                     count += 1
#                     maxcount = max(maxcount, count)
#                 else:
#                     if prev_count is not None and count != prev_count:
#                         print('WARNING: Potential pipeline error')
#                     prev_count = count
#                     count = 1
#                     maxcount = max(maxcount, count)
#                     prev_shr = block.is_shr
#         print(f'stages = {maxcount}')

#         pipe = 0
#         for (i, block) in enumerate(self.blocks):
#             btype = 'Producer' if block.is_shr else 'Consumer'
#             pipe = pipe + 1 if block.is_shr else pipe - 1

#             print(f'{i}: pipe = {pipe}, {btype}')
#             # Prevent re-ordering
#             block.can_dep = False

#             if block.is_shr:
#                 for line in block.lines:
#                     if len(line.src_ptr) == 1:
#                         src = f'cuda::memcpy_async(tile1, {line.dst_ptr}, {line.src_ptr[0]}, cuda::aligned_size_t<{self.dbytes}>(sizeof({self.dtype})), pl);'
#                         line.line = src
#                 src = 'pl.producer_acquire();'
#                 line = SourceLine(src, 0)
#                 block.appendleft(line)
#                 src = 'pl.producer_commit();'
#                 line = SourceLine(src, 0)
#                 block.append(line)
                
#             else:
#                 src = 'pl.consumer_wait();'
#                 line = SourceLine(src, 0)
#                 block.appendleft(line)
#                 src = 'pl.consumer_release();'
#                 line = SourceLine(src, 0)
#                 block.append(line)

#     def _print_blocks(self):
#         source = ""
#         for (i, block) in enumerate(self.blocks):
#             source += f'\n // SourceBlock {i}, shared={block.is_shr}\n' + '\n'.join(line.line for line in block.lines)
#         return source

