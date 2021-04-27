# -*- coding: utf-8 -*-

import re

class LoadOptimisation(object):
    def __init__(self, block_config, var_name):
        
        self.block_config = block_config
        self.var_name = var_name

    def apply(self, source):
        source_copy = source

        pattern = rf'\s{self.var_name}\[.*\];'

        glb_occurance = dict.fromkeys(re.findall(pattern, source), 0)

        sorted_glb_occurance = sorted(glb_occurance.items(), key=lambda x: x[1], reverse=True)

        L1_acc = 0
        for (g, n) in sorted_glb_occurance:
            if n == 1:
                ld_mode = 'lu'
            elif L1_acc < self.block_config.L1_per_thread:
                L1_acc += 1
                ld_mode = 'ca'
            else:
                ld_mode = 'cg'
            addr = ((re.search(r'\[.*\]', g)).group(0))[1:-1]
            glb_occurance[g] = (n, ld_mode, addr)

            dg = f'__ld{ld_mode}({self.var_name} + {addr});'
            source_copy = source_copy.replace(g, ldg)

    return source_copy