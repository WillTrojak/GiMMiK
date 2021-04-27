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
*/


#define SOA_IDX(i, v) ((((i) / ${soasz})*${nvars} + (v))*${soasz} + (i) % ${soasz})

#define ELEM_K(t) (((t) % ${bcfg.warp_size}) % ${p})
#define WARP(t) ((t)/${bcfg.warp_size})
#define ATHRD (${bcfg.warp_size} - ${bcfg.warp_size} % ${p})
#define ABS_E(t, b, bx) ((b)*(ATHRD / ${p})*(bx / 32) + WARP(t)*(ATHRD / ${p}) + ((t) % ${bcfg.warp_size}) / ${p})
#define LCL_E(t) (WARP(t)*(ATHRD / ${p}) + ((t) % ${bcfg.warp_size}) / ${p})
#define W_MASK(t) (((1 << ${p}) - 1) << (${p}*(((t) % ${bcfg.warp_size}) / ${p})))

__global__ void
${funcn}(int n,
         const ${dtype}* __restrict__ ${mem.glb[0].name}, int ${mem.glb[0].ld},
         ${dtype}* __restrict__ ${mem.glb[1].name}, int ${mem.glb[1].ld})
{

    // Memory common 

    // Element No. in global frame
    int ${mem.glb[0].elem} = ((threadIdx.x % ${bcfg.warp_size}) < ATHRD) ? ABS_E(threadIdx.x, blockIdx.x, blockDim.x) : n;
    // Shared Memory Offset
    int ${mem.share[0].offset} = ${bcfg.shr_size_elem}*LCL_E(threadIdx.x);
    // Warp sync mask
    unsigned mask = W_MASK(threadIdx.x);
    // z-plane evaluated by thread 
    int ${mem.glb[0].thrd_v} = ELEM_K(threadIdx.x);  

    ${dtype} ze = ${0.};

    extern __shared__ ${dtype} ${mem.share[0].name}[];

    ${dtype} ${mem.acc_reg.name}[${mem.acc_reg.size}];
    ${dtype} ${mem.local[0].name}[${mem.local[0].size}];
    ${dtype} ${mem.local[1].name}[${mem.local[1].size}];

    // Load differentiation chars into memory
    ${matrix.c_register(A=A, name='d')};

    // n is now number of elements
    if (eg < n)
    {
        ${gen.plane_method_3d(A=A, p=p, nvars=nvars, ndims=ndims, flux_func=flux_n, 
                              block_config=bcfg, mem=mem, thrd_v=mem.glb[0].thrd_v,
                              fargs=fargs, ldst_opt=ldst_opt)}
    }
}

#undef SOA_IDX
#undef ELEM_K
#undef WARP
#undef ATHRD
#undef ABS_E
#undef LCL_E
#undef W_MASK
