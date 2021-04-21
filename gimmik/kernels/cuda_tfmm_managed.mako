# -*- coding: utf-8 -*-
<%namespace module='gimmik.flux2' name='flux'/>
<%namespace module='gimmik.generator' name='gen'/>
<%namespace module='gimmik.matrix' name='matrix'/>
<%namespace module='gimmik.utils' name='utils'/>

/* CHANGE LOG
    Same as flux7 but with:
    * global read for x-line intersection with z-y plane removed.
*/ 

#define AOSOA_V(i) ((((i) - ((i) % ${soasz}))) / ${soasz}) % ${nvars}
#define SOA_IDX(i, v) ((((i) / ${soasz})*${nvars} + (v))*${soasz} + (i) % ${soasz})
#define TSOA2_IDX(j, yo, eo) ( ${nvars}*((j) + (yo)) + (eo))


#define A_THRDS_PER_W (${p*int(warp_size/p)})
#define THRD_IN_W(t) ((t) % ${warp_size})
#define W_NUM(t) ((t) / ${warp_size})
#define W_PER_BLK(bx) ( (bx) / ${warp_size} )
#define ATHRDS_PER_BLK(bx) (W_PER_BLK(bx) * ${p*int(warp_size/p)})
#define ELEM_PER_BLK(bx) ((W_PER_BLK(bx) * ${p*int(warp_size/p)}) / ${p})

#define E_NUM_L(t) ((W_NUM(t)*${p*int(warp_size/p)} + THRD_IN_W(t)) / ${p})
#define E_NUM(b, bx, t) ((b)*ELEM_PER_BLK(bx) + E_NUM_L(t))

#define ELEM_K(t) (((t) % ${warp_size}) % ${p})


__global__ void
${funcn}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{

    // Memory common 
    //int eg = (THRD_IN_W(threadIdx.x) < ${p*int(warp_size/p)}) ? E_NUM(threadIdx.x, blockDim.x, blockIdx.x) : n; // element No. in global frame
    //int el_offset = ${shr_size_elem}*E_NUM_L(threadIdx.x);

    int eg = (blockIdx.x*blockDim.x + threadIdx.x)/${p}; 
    int el_offset = ${p*p*nvars}*(threadIdx.x/${p});

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

#undef AOSOA_V
#undef SOA_IDX
#undef TSOA2_IDX

#undef ELEM_NUM
#undef ELEM_GLB_O
#undef ELEM_NUM_L
#undef ELEM_K