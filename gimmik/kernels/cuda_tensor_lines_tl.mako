# -*- coding: utf-8 -*-

/* CHANGE LOG
    p*p threads, with each thread doing lines and all varaibles
    each thread participates in the global load 
*/ 

/*
    NDIMS              = ${ndims}
    ORDER (P)          = ${p-1} (${p})
    NVARS              = ${nvars}
    ELEMENTS PER BLOCK = ${block_elem}

    SHARED USED [B]    = ${shr_used}
    DATA TYPE          = ${dtype}
    SOA_SZ             = ${soasz}
    SHARED SOA_SZ      = ${ssoasz}
    BLOCK CONFIG       = (${p}, ${p}, ${block_elem})
*/


#include <cooperative_groups.h>

#define SOA_IDX(i, v) ((((i) / ${soasz})*${nvars} + (v))*${soasz} + (i) % ${soasz})
#define SSOA_IDX(i, v) ((((i) / ${ssoasz})*${nvars} + (v))*${ssoasz} + (i) % ${ssoasz})

namespace cg = cooperative_groups;

__constant__ ${dtype} d[${p*p}] = {${','.join(str(D[j,i]) for j in range(p) for i in range(p))} };

__device__ void
${funcn}_flux(int el,
              ${dtype} const d,
              ${dtype}* __restrict__ u,
              ${dtype} const jac0,
              ${dtype} const jac1,
              ${dtype} const jac2,
              ${dtype}* dotp
             )
{
    ${dtype} ur[${nvars}];
    % for v in range(nvars):
        ur[${v}] = u[SSOA_IDX(el, ${v})]; 
    % endfor

    dotp[ 4] += d*jac0*(${-1/c['tr']}*ur[1]);
    dotp[ 5] += d*jac1*(${-1/c['tr']}*ur[1]);
    dotp[ 6] += d*jac2*(${-1/c['tr']}*ur[1]);
    dotp[ 7] += d*jac0*(${-1/c['tr']}*ur[2]);
    dotp[ 8] += d*jac1*(${-1/c['tr']}*ur[2]);
    dotp[ 9] += d*jac2*(${-1/c['tr']}*ur[2]);
    dotp[10] += d*jac0*(${-1/c['tr']}*ur[3]);
    dotp[11] += d*jac1*(${-1/c['tr']}*ur[3]);
    dotp[12] += d*jac2*(${-1/c['tr']}*ur[3]);
    dotp[ 0] += d*${c['zeta']}*(jac0*ur[1] + jac1*ur[2] + jac2*ur[3]);
    dotp[ 1] += d*(jac0*(ur[1]*ur[1] - ${c['nu']}*ur[4] + ur[0]) + 
                   jac1*(ur[1]*ur[2] - ${c['nu']}*ur[5]) +
                   jac2*(ur[1]*ur[3] - ${c['nu']}*ur[6]));
    dotp[ 2] += d*(jac0*(ur[2]*ur[1] - ${c['nu']}*ur[7]) + 
                   jac1*(ur[2]*ur[2] - ${c['nu']}*ur[8] + ur[0]) +
                   jac2*(ur[2]*ur[3] - ${c['nu']}*ur[9]));
    dotp[ 3] += d*(jac0*(ur[3]*ur[1] - ${c['nu']}*ur[10]) + 
                   jac1*(ur[3]*ur[2] - ${c['nu']}*ur[11]) +
                   jac2*(ur[3]*ur[3] - ${c['nu']}*ur[12] + ur[0]));
}

__device__ void
${funcn}_mul(int k, int eg, int el,
             ${dtype}* __restrict__ b, int ldb,
             ${dtype}* __restrict__ c, int ldc
            )
{
    __builtin_assume(ldb > 0);
    __builtin_assume(ldc > 0);

    int i  = threadIdx.x;
    int j  = threadIdx.y;

    ${dtype} jacx[] = {1., 0., 0.}, jacy[] = {0., 1., 0.}, jacz[] = {0., 0., 1.};

    ${dtype} dotp[] = {${','.join(str(0.) for i in range(nvars))}};

    // x-line
    for(int x=0; x<${p}; x++)
    {
        ${funcn}_flux(el, d[x*${p} + j],
                      b + (x + j*${p} + k*${p*p})*ldb,
                      jacx[0], jacx[1], jacx[2], dotp);
    }

    // y-line
    for(int x=0; x<${p}; x++)
    {        
        ${funcn}_flux(el, d[x*${p} + i],
                      b + (i + x*${p} + k*${p*p})*ldb,
                      jacy[0], jacy[1], jacy[2], dotp);
    }

    // z-line
    for(int x=0; x<${p}; x++)
    {
        ${funcn}_flux(el, d[x*${p} + k],
                      b + (i + j*${p} + x*${p*p})*ldb,
                      jacz[0], jacz[1], jacz[2], dotp);
    }

    for (int v=0; v<${nvars}; v++)
        c[SOA_IDX(eg, v) + (i + j*${p} + k*${p*p})*ldc] = dotp[v];
    return;
}

__device__ void
${funcn}_import(int eg, int el,
                ${dtype}* __restrict__ g, int ldg,
                ${dtype}* __restrict__ s, int lds)
{

    __builtin_assume(ldg > 0);
    __builtin_assume(lds > 0);

    int i  = threadIdx.x;
    int j  = threadIdx.y;

    for(int k=0; k<${p}; k++)
    {    
    % for v in range(nvars):
        //s[SSOA_IDX(el, ${v}) + (i + j*${p} + k*${p*p})*lds] = __ldg(g + SOA_IDX(eg, ${v}) + (i + j*${p} + k*${p*p})*ldg);
        s[SSOA_IDX(el, ${v}) + (i + j*${p} + k*${p*p})*lds] = g[SOA_IDX(eg, ${v}) + (i + j*${p} + k*${p*p})*ldg];
    % endfor
    }

    return;
}

__global__ void
__launch_bounds__ (${block_elem*p*p})
${funcn}(int n,
         ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
    __builtin_assume(ldb > 0);
    __builtin_assume(ldc > 0);

    auto block = cg::this_thread_block();

    // Element No. in global frame
    int el = threadIdx.z;
    int eg = (el < ${block_elem}) ? (blockIdx.x*${block_elem} + el) : n;

    extern __shared__ ${dtype} s[];
    int lds = ${block_elem*nvars};
    
    // Import from global to shared
    if(eg < n)
        ${funcn}_import(eg, el, b, ldb, s, lds);
    __syncthreads();
    
    // Calculate the flux and matmul.
    if(eg < n)
        for (int k=0; k<${p-1}; k++)
            ${funcn}_mul(k, eg, el, s, lds, c, ldc);

}