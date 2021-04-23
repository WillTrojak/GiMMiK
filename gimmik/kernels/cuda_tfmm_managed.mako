# -*- coding: utf-8 -*-
<%namespace module='gimmik.generator' name='gen'/>
<%namespace module='gimmik.matrix' name='matrix'/>

/* CHANGE LOG
   Methodology of planar 3d, but with managed memory
*/ 

#define SOA_IDX(i, v) ((((i) / ${soasz})*${nvars} + (v))*${soasz} + (i) % ${soasz})

#define ELEM_K(t) (((t) % ${warp_size}) % ${p})
#define WARP(t) ((t)/${warp_size})
#define ATHRD (${warp_size} - ${warp_size} % ${p})
#define ABS_E(t, b, bx) ((b)*(ATHRD / ${p})*(bx / 32) + WARP(t)*(ATHRD / ${p}) + ((t) % ${warp_size}) / ${p})
#define LCL_E(t) (WARP(t)*(ATHRD / ${p}) + ((t) % ${warp_size}) / ${p}) 

__global__ void
${funcn}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{

    // Memory common 
    int eg = ((threadIdx.x % ${warp_size}) < ATHRD) ? ABS_E(threadIdx.x, blockIdx.x, blockDim.x) : n; // element No. in global frame
    int el_offset = ${shr_size_elem}*LCL_E(threadIdx.x);

    ${dtype} ze = ${0.};

    //__shared__ ${dtype} bs[${shr_size}];

    extern __shared__ ${dtype} bsd[];

    ${dtype} accl[${nvars}], bly[${nvars*p}], blx[${nvars}];

    // Load differentiation chars into memory
    ${matrix.c_register(A=A, name='d')};

    // n is now number of elements
    if (eg < n)
    {
        int thrd_k = ELEM_K(threadIdx.x);  // z-plane evaluated by thread

        ${gen.plane_method_3d(A=A, elem='eg', p=p, nvars=nvars, flux_func=flux_n, 
                              shr_size=shr_size_elem,
                              thrd_v='thrd_k', ldi='ldb', ldo='ldc',
                              macro_name='SOA_IDX', shr_offset='el_offset',
                              fargs={'a': [1,1,1], 'ndims': ndims})}
    }
}

#undef SOA_IDX
#undef ELEM_K
#undef WARP
#undef ATHRD
#undef ABS_E
#undef LCL_E