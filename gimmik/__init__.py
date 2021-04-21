# -*- coding: utf-8 -*-

import pkgutil
import math
import re
import warnings

from mako.template import Template
import numpy as np

from gimmik._version import __version__

import gimmik.euler
import gimmik.fluxcls
import gimmik.flux2
import gimmik.generator
import gimmik.hyperns
import gimmik.linear
from gimmik.matrix import GimmikMatrix, c_register
import gimmik.methods
import gimmik.utils

class GimmikConfig:

    def __init__(self, platform, dtype, maxlen=None):
        
        # Platform language base
        self._pltfs = {'c': 'c',
                       'c_cblank': 'c',
                       'c-omp': 'c',
                       'cuda': 'c',
                       'cuda_blanking': 'c',
                       'cuda_fblanking': 'c',
                       'cuda_tensor_flux': 'c',
                       'cuda_tensor_flux2': 'c',
                       'cuda_tensor_flux2a': 'c',
                       'cuda_tensor_flux3': 'c',
                       'cuda_tensor_flux4': 'c',
                       'cuda_tensor_flux5': 'c',
                       'cuda_tensor_flux6': 'c',
                       'cuda_tensor_flux7': 'c',
                       'cuda_tensor_flux8': 'c',
                       'cuda_tensor_flux9': 'c',
                       'cuda_tensor_flux10': 'c',
                       'cuda_tensor_flux11': 'c',
                       'cuda_tensor_single': 'c',
                       'cuda_tfmm_managed': 'c',
                       'ispc': 'c',
                       'opencl': 'c',
                       'f90T': 'f90',
                       'f90T-cuda': 'f90'}

        self._types = {'c'  : {np.float32: ( 'float','f'),
                               np.float64: ('double', '')},
                       'f90': {np.float32: ('real(kind=4)','_4'),
                               np.float64: ('real(kind=8)','_8')}}

        self._type_size = {np.float64: 8,
                           np.float32: 4}

        # Language continuation character
        self._cchar = {'c' : ' ',
                      'f90': '&'}

        self.platform = platform
        try:
            self.lang = self._pltfs[platform]
        except KeyError:
            raise KeyError(f"GiMMiK: platform '{platform}' not recognised")

        self.cchar = self._cchar[self.lang]
        self.maxlen = maxlen
        
        self.dtype = np.dtype(dtype).type
        self.bytes = self._type_size[dtype]

        # np type to language specific types
        try:
            (self.dtype, self.suffix) = self._types[self.lang][dtype]
        except KeyError:
            raise ValueError('GiMMiK: Invalid floating point data type')
        
    def cleanup(self, src):
        # Append suffix to handle typing
        src = re.sub(r'(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',
                     rf'\g<0>{self.suffix}', src)

        # Split lines to enforce line length max (needed for F90-F08 ISO)
        if self.maxlen:
            src = _line_split(self.maxlen, self.cchar, src)
        elif self.lang == 'f90':
            warnings.warn('GiMMiK: No maxlen given for F90 based kernel')

        return src

    def _line_split(maxlen, cchar, src):
        lines = src.splitlines()

        src = ''
        for line in lines:
            nidnt = len(line) - len(line.lstrip(' '))
            
            while math.ceil(len(line)/maxlen) > 1:
                ns = max(line[:maxlen].rfind('+ '),line[:maxlen].rfind('- '))
                src += line[:ns] + cchar + '\n'
                line = nidnt*' ' + line[ns:]

            src += line + '\n'

        return src

        
def generate_mm(mat, dtype, platform, alpha=1.0, beta=0.0, funcn='gimmik_mm',
                maxlen=None):
    support = {'c', 'c-omp', 'cuda', 'ispc', 'opencl', 'f90T', 'f90T-cuda'}
    _issuported(platform, support, funcn)

    cfg = GimmikConfig(platform, dtype, maxlen)

    # Multiply the matrix through by alpha
    mat = alpha*mat

    # Template arguments
    tplargs = {'dtype': cfg.dtype, 'mat': mat, 'beta': beta, 'funcn': funcn}

    # Load and render the template
    tpl = pkgutil.get_data(__name__, f'kernels/{platform}.mako')
    src = Template(tpl).render(**tplargs)

    return cfg.cleanup(src)


def generate_smm(mat, dtype, platform, nvars, soasz, skip, alpha=1.0, beta=0.0,
                 funcn='gimmik_smm', maxlen=None):
    
    cfg = GimmikConfig(platform, dtype, maxlen)

    # Multiply the matrix through by alpha
    mat = alpha*mat

    # Template arguments
    tplargs = {'dtype': cfg.dtype, 'mat': mat, 'beta': beta, 'nvars': nvars,
               'soasz': soasz, 'skip': skip, 'funcn': funcn}

    # Load and render the template
    tpl = pkgutil.get_data(__name__, f'kernels/{platform}.mako')
    src = Template(tpl).render(**tplargs)
    
    return cfg.cleanup(src)


def generate_bmm(mat, dtype, platform, nvars, ndims, soasz, not_blanked, alpha=1.0, beta=0.0,
                 funcn='gimmik_bmm', maxlen=None):
    support = {'cuda_fblanking'}
    _issuported(platform, support, funcn)

    cfg = GimmikConfig(platform, dtype, maxlen)

    # Multiply the matrix through by alpha
    mat = alpha*mat

    (n, k) = np.shape(mat)
    # Template arguments
    tplargs = {'dtype': cfg.dtype, 'mat': mat, 'beta': beta, 'nvars': nvars, 'ndims': ndims,
               'soasz': soasz, 'not_blanked': not_blanked, 'np': n, 'funcn': funcn}

    # Load and render the template
    tpl = pkgutil.get_data(__name__, f'kernels/{platform}.mako')
    src = Template(tpl).render(**tplargs)
    
    return cfg.cleanup(src)

def generate_tfmm(D, ndims, nvars, dtype, block_dim, soasz, platform, flux,
                  funcn='gimmik_tfmm', maxlen=None):

    cfg = GimmikConfig(platform, dtype, maxlen)

    (p, k) = np.shape(D)
    # Template arguments
    tplargs = {'D': D, 'ndims': ndims, 'p': p, 'nvars': nvars, 
               'blk_dim': block_dim, 'soasz': soasz,
               'dtype': cfg.dtype, 'flux_n': flux, 'funcn': funcn}

    # Load and render the template
    tpl = pkgutil.get_data(__name__, f'kernels/{platform}.mako')
    src = Template(tpl).render(**tplargs)

    return cfg.cleanup(src)

def generate_tfmm_managed(mat, ndims, nvars, dtype, block_dim, soasz, platform,
                  flux, shared_max=None, warp_size=32, funcn='gimmik_tfmm', maxlen=None,
                  tplargs=None):
    if tplargs is None:
        tplargs = {}

    #if platform.find('cuda') == -1:
    #    raise ValueError(f'Gimmik: tfmm managed only supported with CUDA platforms')

    cfg = GimmikConfig(platform, dtype, maxlen)

    (p, k) = np.shape(mat)
    A = GimmikMatrix(mat, cfg.dtype)

    warps = math.floor(block_dim/warp_size)
    active_threads = warps*p*math.floor(warp_size/p)
    print(warps,active_threads)

    if shared_max is None:
        shared_size = p*nvars*active_threads
    else:
        if shared_max < cfg.bytes*p*active_threads*nvars:
            raise ValueError(f'GiMMiK: insufficent shared memory given')
        shared_size = math.floor(shared_max/cfg.bytes)

    shared_size_elem = math.floor(p*shared_size/active_threads)

    tplargs.update({'A': A, 'ndims': ndims, 'p': p, 'nvars': nvars, 
                    'shr_size': shared_size, 
                    'shr_size_elem': shared_size_elem,
                    'soasz': soasz, 'warp_size': warp_size, 'dtype': cfg.dtype, 
                    'flux_n': flux, 'funcn': funcn})

    tpl = pkgutil.get_data(__name__, f'kernels/{platform}.mako')
    src = Template(tpl).render(**tplargs)

    return cfg.cleanup(src)

def _issuported(platform, support, funcn):
    if platform not in support:
        raise ValueError(f'GiMMiK: {platform} unsupported in {funcn}')
