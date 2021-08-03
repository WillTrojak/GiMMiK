# -*- coding: utf-8 -*-

import numpy as np
from mako.template import Template
from math import ceil
import pkgutil
import re

from gimmik._version import __version__
from gimmik.test import default_cfg, get_tester


class GimmikConfig(object):
    def __init__(self, platform, dtype, maxlen=None):

        self._types = {np.float32: ( 'float','f'), np.float64: ('double', '')}

        self._type_size = {np.float64: 8, np.float32: 4}

        self.platform = platform

        self.cchar = ''
        self.maxlen = maxlen
        
        self.dtype = np.dtype(dtype).type
        self.bytes = self._type_size[dtype]

        # np type to language specific types
        try:
            (self.dtype, self.suffix) = self._types[dtype]
        except KeyError:
            raise ValueError('GiMMiK: Invalid floating point data type')
        
    def cleanup(self, src):
        # Append suffix to handle typing
        src = re.sub(r'(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',
                     rf'\g<0>{self.suffix}', src)

        # Split lines to enforce line length max (needed for F90-F08 ISO)
        if self.maxlen is not None:
            src = self._line_split(src)

        return src

    def _line_split(self, src):
        lines = src.splitlines()

        src = ''
        for line in lines:
            nidnt = len(line) - len(line.lstrip(' '))
            
            while ceil(len(line)/self.maxlen) > 1:
                ns = max(line[:self.maxlen].rfind('+ '),
                         line[:self.maxlen].rfind('- '))
                src += line[:ns] + self.cchar + '\n'
                line = nidnt*' ' + line[ns:]

            src += line + '\n'

        return src


def generate_mm(mat, dtype, platform, alpha=1.0, beta=0.0, funcn='gimmik_mm',
                maxlen=None, block_dim=None):
    
    cfg = GimmikConfig(platform, dtype, maxlen)

    # Multiply the matrix through by alpha
    mat = alpha*mat

    # Template arguments
    tplargs = {'dtype': dtype, 'mat': mat, 'beta': beta, 'funcn': funcn,
               'block_dim': block_dim}

    # Load and render the template
    tpl = pkgutil.get_data(__name__, 'kernels/{0}.mako'.format(platform))
    src = Template(tpl).render(**tplargs)

    # Return the source
    return cfg.cleanup(src)

def generate_mm_split(mat, dtype, platform, block_dim, split, alpha=1.0,
                      beta=0.0, funcn='gimmik_mm', maxlen=None):

    cfg = GimmikConfig(platform, dtype, maxlen)

    # Multiply the matrix through by alpha
    mat = alpha*mat

    # Split config
    row_per_warp = int(np.shape(mat)[0]/split)

    # Template arguments
    tplargs = {'dtype': dtype, 'mat': mat, 'beta': beta, 'funcn': funcn,
               'block_dim': block_dim, 'row_per_warp': row_per_warp,
               'split': split}

    # Load and render the template
    tpl = pkgutil.get_data(__name__, 'kernels/{0}.mako'.format(platform))
    src = Template(tpl).render(**tplargs)

    return cfg.cleanup(src)

def profile_generated(mat, dtype, src, platform, **kwargs):
    cfg = default_cfg(dtype, **kwargs)
    tester = get_tester(platform, cfg)

    return tester.mul_profile(src, mat)

def profile_cublas(mat, dtype, alpha=1., beta=0.):
    cfg = default_cfg(dtype)

    tester = get_tester('cuda', cfg)
    return tester.mul_cublas_profile(mat, alpha, beta)

def profile_rocblas(mat, dtype, alpha=1., beta=0.):
    cfg = default_cfg(dtype)

    tester = get_tester('hip', cfg)
    return tester.mul_rocblas_profile(mat, alpha, beta)

def optimise_block_dim(mat, dtype, src, platform, max_size=1024):
    if platform not in ['cuda', 'hip']:
        raise ValueError('Invalid platform for block size optimisation')
    
    cfg = default_cfg(dtype)
    tester = get_tester(platform, cfg)
    soasz = tester.backend.soasz

    runtime = []
    for i in range(1, int(max_size/soasz)):
        threads = i*soasz

        tester.cfg.set('gimmik-profile', 'block_dim', threads)
        runtime.append(tester.mul_profile(src, mat)['runtime'])

    return (runtime.index(min(runtime)) + 1)*soasz
