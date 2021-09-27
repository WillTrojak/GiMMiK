# -*- coding: utf-8 -*-
<%namespace module='gimmik.hyperns' name='flux'/>

/* CHANGE LOG
    p*p threads, with each thread doing lines and all varaibles
    each thread participates in the global load,
    fully unrolled
*/ 

/*
    NDIMS              = ${ndims}
    ORDER (P)          = ${p-1} (${p})
    NVARS              = ${nvars}
    ELEMENTS PER BLOCK = ${block_elem}

    SHARED USED [B]    = ${shr_size}
    DATA TYPE          = ${dtype}
    SOA_SZ             = ${soasz}
    BLOCK CONFIG       = (${block_elem*p*p}, 1, 1)
*/

#include <cooperative_groups.h>
#include <cuda/barrier>
namespace cg = cooperative_groups;

#define SOA_SZ ${soasz}
#define SOA_IDX(i, v) ((((i) / ${soasz})*${nvars} + (v))*${soasz} + (i) % ${soasz})
#define EPB ${block_elem}
#define SSOA_SZ ${block_elem}
#define SHR_NV ${nvars-1-ndims*(nvars-1)}
#define ACC_NV ${nvars-1}

#define BLOCK_DIMX ${block_elem*p*p}
#define SHR_SIZE ${shr_size}
#define ORDER ${p-1}
#define WTYPE ${dtype}
#define N_VAR ${nvars}

#define SHR_IDX_I(e, v, i, j, k) ( ((e)/SSOA_SZ)*(${p*p*p}*SHR_NV*SSOA_SZ) + (e)%SSOA_SZ + ((i) + (j)*${p} + (k)*${p*p})*SSOA_SZ + (v)*SSOA_SZ*${p*p*p})
#define ACC_IDX_I(e, v, i, j, k) ( ((e)/SSOA_SZ)*(${p*p*p}*ACC_NV*SSOA_SZ) + (e)%SSOA_SZ + ((i) + (j)*${p} + (k)*${p*p})*SSOA_SZ + (v)*SSOA_SZ*${p*p*p})


__constant__ ${dtype} dc[${p*p}] = {${','.join(str(D[j,i]) for j in range(p) for i in range(p))} };

__global__ void
__launch_bounds__ (${block_elem*p*p})
${funcn}(int n,
        ${dtype}* __restrict__ b, int ldb,
        ${dtype}* __restrict__ c, int ldc
       )
{
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__  barrier bar;
    auto block = cg::this_thread_block();

    extern __shared__ ${dtype} s[];

    int i = (threadIdx.x/${block_elem}) % ${p};
    int j = (threadIdx.x/${block_elem*p}) % ${p};
    int k = threadIdx.x % ${block_elem};
  
    int oss = 0;
    int osa = ${block_elem*(nvars-1)*p*p*p};
    int el = k;
    int eg = blockIdx.x*${block_elem} + el;

    ${dtype} jacx[] = {1.,0.,0.};
    ${dtype} jacy[] = {0.,1.,0.};
    ${dtype} jacz[] = {0.,0.,1.};

    ${dtype} dotp[${nvars}];

    if (block.thread_rank() == 0)
        init(&bar, (min((blockIdx.x+1)*${block_elem}, n) - blockIdx.x*${block_elem})*${p*p});
    block.sync();

    if(eg < n)
    {
    % for k in range(p):
    
    % for i in range(p):
    % for j in range(p):
    % for v in range(nvars):
        ${flux.acmhd_24_import(v, 'b', 'ldb', 'dotp', 's', c, p, i, j, k)}
    % endfor
    % endfor
    % endfor
        bar.arrive_and_wait();

    % for v in range(nvars-1):
        dotp[0] = ${'+'.join('{f}'.format(f=flux.acmhd_24_acc(v, 's', f'dc[{x*p} + i]', 'jacx', x, 'j', k)) for x in range(p))};
        s[osa + ACC_IDX_I(el, ${v}, i, j, ${k})] = dotp[0] + ${'+'.join('{f}'.format(f=flux.acmhd_24_acc(v, 's', f'dc[{x*p} + j]', 'jacy', 'i', x, k)) for x in range(p))};
    % endfor

    % endfor
        bar.arrive_and_wait();
    
    % for k in range(p):
    % for v in range(nvars-1):
        dotp[${v}] = ${'+'.join('{f}'.format(f=flux.acmhd_24_acc(v, 's', f'dc[{x*p} + {k}]', 'jacz', 'i', 'j', x)) for x in range(p))};
    % endfor

        b[SOA_IDX(eg, 0) + (i + j*${p} + ${k*p*p})*ldb] = ${c['ac-zeta']}*(dotp[3] + dotp[7] + dotp[11] + s[osa + ACC_IDX_I(el, 3, i, j, ${k})] + s[osa + ACC_IDX_I(el, 7, i, j, ${k})] + s[osa + ACC_IDX_I(el, 11, i, j, ${k})]);
    % for v in range(ndims):
        b[SOA_IDX(eg, ${v+1}) + (i + j*${p} + ${k*p*p})*ldb] = dotp[${v}] + s[osa + ACC_IDX_I(el, ${v}, i, j, ${k})];
    % endfor
    % for v in range(ndims*ndims):
        b[SOA_IDX(eg, ${v+4}) + (i + j*${p} + ${k*p*p})*ldb] = ${-1/c['tr']}*(dotp[${v+3}] + s[osa + ACC_IDX_I(el, ${v+3}, i, j, ${k})]);
    % endfor
    % endfor
    }
    
  return;
}

#undef SOA_IDX
#undef SHR_IDX_I
#undef ACC_IDX_I
#undef ACC_NV
#undef SHR_NV
#undef SSOA_SZ