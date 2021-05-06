# -*- coding: utf-8 -*-
<%namespace module='gimmik.generator' name='gen'/>
<%namespace module='gimmik.matrix' name='matrix'/>

/* CHANGE LOG
   Methodology of planar 3d, but with managed memory
*/ 

/* CONFIG DETAILS
  DATA TYPE             = ${dtype}

  P (POINTS PER LINE)   = ${p-1} (${p})
  NVARS                 = ${nvars}
  FLUX                  = ${flux_n}
  
  SOASZ                 = ${soasz}
  WARP SIZE             = ${bcfg.warp_size}
  TOTAL SHARED SIZE (B) = ${bcfg.shr_size}
  SHARED VALS PER ELEM  = ${bcfg.shr_size_elem}
  USAGE SHR PER ELEM    = ${bcfg.usable_shr_elem}

  LOAD OPT              = ${opargs['ld_opt']}
  STORE OPT             = ${opargs['st_opt']}
  COMPUTE INTERLEAVING  = ${opargs['intl_opt']}
  OP SHR ORDER          = ${opargs['shr_op_order']}
  SHR BANK DE-CONFLICT  = ${opargs['shr_bdc']} 
*/

#include <stdio.h>

#define SOA_IDX(i, v) ((((i) / ${soasz})*${nvars} + (v))*${soasz} + (i) % ${soasz})

#define ELEM_K(t) (((t) % ${bcfg.warp_size}) % ${p})
#define WARP(t) ((t)/${bcfg.warp_size})
#define ATHRD (${bcfg.warp_size} - ${bcfg.warp_size} % ${p})
#define ABS_E(t, b, bx) ((b)*(ATHRD / ${p})*(bx / 32) + WARP(t)*(ATHRD / ${p}) + ((t) % ${bcfg.warp_size}) / ${p})
#define LCL_E(t) (WARP(t)*(ATHRD / ${p}) + ((t) % ${bcfg.warp_size}) / ${p})
#define WRP_E(t) (((t) % ${bcfg.warp_size} ) / ${p})
#define W_MASK(t) (((1 << ${p}) - 1) << (${p}*(((t) % ${bcfg.warp_size}) / ${p})))

// Load differentiation chars into memory
    ${matrix.c_register(A=A, name='d')};

__constant__ ${dtype} ze = ${0.};

__global__ void
__launch_bounds__ (${bcfg.blk_dim})
${funcn}(int n,
         const ${dtype}* __restrict__ ${mem.glb[0].name}, int ${mem.glb[0].ld},
         ${dtype}* __restrict__ ${mem.glb[1].name}, int ${mem.glb[1].ld})
{

    // Memory common 

    // Element No. in global frame
    int ${mem.glb[0].elem} = ((threadIdx.x % ${bcfg.warp_size}) < ATHRD) ? ABS_E(threadIdx.x, blockIdx.x, blockDim.x) : n;
    // Shared Memory Offset
    % if opargs['shr_bdc']:
    int ${mem.share[0].offset} = ${bcfg.shr_size_elem}*LCL_E(threadIdx.x) +
    ((${bcfg.warp_size} - ((${bcfg.shr_offset_rem}*WRP_E(threadIdx.x))%${bcfg.warp_size})) % ${bcfg.warp_size} + WRP_E(threadIdx.x)*${p}) % ${bcfg.warp_size};
    % else:
        int ${mem.share[0].offset} = ${bcfg.shr_size_elem}*LCL_E(threadIdx.x);
    % endif
    // Warp sync mask
    unsigned mask = W_MASK(threadIdx.x);
    // z-plane evaluated by thread 
    int ${mem.glb[0].thrd_v} = ELEM_K(threadIdx.x);  

% for i in range(mem.n_share):
    extern __shared__ ${dtype} ${mem.share[i].name}[];
% endfor

    ${dtype} ${mem.acc_reg.name}[${mem.acc_reg.size}];
% for i in range(mem.n_local):
    ${dtype} ${mem.local[i].name}[${mem.local[i].size}];
% endfor
    

    // n is now number of elements
    if (eg < n)
    {
        ${gen.plane_method_3d(A=A, p=p, nvars=nvars, ndims=ndims, flux_func=flux_n, 
                              block_config=bcfg, mem=mem, thrd_v=mem.glb[0].thrd_v,
                              fargs=fargs, opargs=opargs)}
    }
}

#undef SOA_IDX
#undef ELEM_K
#undef WARP
#undef ATHRD
#undef ABS_E
#undef LCL_E
#undef WRP_E
#undef W_MASK
