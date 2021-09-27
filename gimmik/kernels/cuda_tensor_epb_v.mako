# -*- coding: utf-8 -*-

/* CHANGE LOG
    Functions split off to reduce register pressure
*/ 

/*
    NDIMS              = ${ndims}
    ORDER (P)          = ${p-1} (${p})
    NVARS              = ${nvars}
    ELEMENTS PER BLOCK = ${block_elem}

    SHARED USED [B]    = ${shr_used}
    DATA TYPE          = ${dtype}
    VECTOR WIDTH       = ${vec_width}
    BLOCK CONFIG       = (${p}, ${p}, ${block_elem})
*/


#include <cooperative_groups.h>
//#include <cooperative_groups/memcpy_async.h>
//#include <cuda/barrier>
//#include <cuda/pipeline>

#define SOA_IDX(i, v) ((((i) / ${soasz})*${nvars} + (v))*${soasz} + (i) % ${soasz})
#define SSOA_IDX(i, v) ((((i) / ${vec_width})*${nvars} + (v))*${vec_width} + (i) % ${vec_width})

namespace cg = cooperative_groups;

__constant__ ${dtype} d[${p*p}] = {${','.join(str(D[j,i]) for j in range(p) for i in range(p))} };

__device__ void
${funcn}_flux(${dtype} d,
            % for v in range(nvars):
                ${dtype}* __restrict__ u_${v},
            % endfor
                ${dtype} const jac0,
                ${dtype} const jac1,
                ${dtype} const jac2,
                ${dtype}* dotp
             )
{

    dotp[ 0] += d*${c['zeta']}*(jac0*u_1[0] + jac1*u_2[0] + jac2*u_3[0]);
    dotp[ 1] += d*(jac0*(u_1[0]*u_1[0] - ${c['nu']}*u_4[0] + u_0[0]) + 
                   jac1*(u_1[0]*u_2[0] - ${c['nu']}*u_5[0]) +
                   jac2*(u_1[0]*u_3[0] - ${c['nu']}*u_6[0]));
    dotp[ 2] += d*(jac0*(u_2[0]*u_1[0] - ${c['nu']}*u_7[0]) + 
                   jac1*(u_2[0]*u_2[0] - ${c['nu']}*u_8[0] + u_0[0]) +
                   jac2*(u_2[0]*u_3[0] - ${c['nu']}*u_9[0]));
    dotp[ 3] += d*(jac0*(u_3[0]*u_1[0] - ${c['nu']}*u_10[0]) + 
                   jac1*(u_3[0]*u_2[0] - ${c['nu']}*u_11[0]) +
                   jac2*(u_3[0]*u_3[0] - ${c['nu']}*u_12[0] + u_0[0]));
    dotp[ 4] += d*jac0*(${-1/c['tr']}*u_1[0]);
    dotp[ 5] += d*jac1*(${-1/c['tr']}*u_1[0]);
    dotp[ 6] += d*jac2*(${-1/c['tr']}*u_1[0]);
    dotp[ 7] += d*jac0*(${-1/c['tr']}*u_2[0]);
    dotp[ 8] += d*jac1*(${-1/c['tr']}*u_2[0]);
    dotp[ 9] += d*jac2*(${-1/c['tr']}*u_2[0]);
    dotp[10] += d*jac0*(${-1/c['tr']}*u_3[0]);
    dotp[11] += d*jac1*(${-1/c['tr']}*u_3[0]);
    dotp[12] += d*jac2*(${-1/c['tr']}*u_3[0]);
}

__device__ void
${funcn}_line(int eg, int el,
              ${dtype}* __restrict__ b, int ldb,
              ${dtype}* __restrict__ c, int ldc
             )
{
    __builtin_assume(ldb > 0);
    __builtin_assume(ldc > 0);

    int i  = threadIdx.x;
    int j  = threadIdx.y;

    ${dtype} jacx[] = {1., 0., 0.}, jacy[] = {0., 1., 0.}, jacz[] = {0., 0., 1.};

    for(int k=0; k<${p}; k++)
    {
        ${dtype} dotp[] = {${','.join(str(0.) for i in range(nvars))}};
        for(int x=0; x<${p}; x++)
        {
            ${funcn}_flux(d[x*${p} + j],
            % for v in range(nvars):
                b + SSOA_IDX(el, ${v}) + (x + j*${p} + k*${p*p})*ldb,
            % endfor
                jacx[0], jacx[1], jacx[2], dotp);

            ${funcn}_flux(d[x*${p} + i],
            % for v in range(nvars):
                b + SSOA_IDX(el, ${v}) + (i + x*${p} + k*${p*p})*ldb,
            % endfor
                jacy[0], jacy[1], jacy[2], dotp);

            ${funcn}_flux(d[x*${p} + k],
            % for v in range(nvars):
                b + SSOA_IDX(el, ${v}) + (i + j*${p} + x*${p*p})*ldb,
            % endfor
                jacz[0], jacz[1], jacz[2], dotp);
        }

        for (int v=0; v<${nvars}; v++)
            c[SSOA_IDX(el, v) + (i + j*${p} + k*${p*p})*ldc] = dotp[v];
    }
}

__device__ void
${funcn}_import(${dtype}* __restrict__ b_glb, int ldb,
                ${dtype}* __restrict__ b_shr)
{
    __builtin_assume(ldb > 0);

    int el0 = threadIdx.z*${vec_width};
    int e0  = blockIdx.x*${block_elem} + el0;

    int os = threadIdx.x + threadIdx.y*${p};

    for (int k=0; k<${p}; k++)
    {
        for (int v=0; v<${nvars}; v++)
        {
            //*((${dtype}${vec_width} *) &b_shr[SSOA_IDX(el0, v) + (os + k*${p*p})*${nvars*vec_width}]) = *(${dtype}${vec_width}*)(static_cast <${dtype} *>(__builtin_assume_aligned(b_glb + SOA_IDX(e0, v) + (os + k*${p*p})*ldb, 64)));

            *((${dtype}${vec_width} *) &b_shr[SSOA_IDX(el0, v) + (os + k*${p*p})*${nvars*vec_width}]) = *(${dtype}${vec_width}*)(b_glb + SOA_IDX(e0, v) + (os + k*${p*p})*ldb);
        }
    }
    return;
}

__device__ void
${funcn}_export(${dtype}* __restrict__ b_shr,
                ${dtype}* __restrict__ b_glb, int ldb)
{
    __builtin_assume(ldb > 0);

    int el0 = threadIdx.z*${vec_width};
    int e0  = blockIdx.x*${block_elem} + el0;

    int os = threadIdx.x + threadIdx.y*${p};

    for (int k=0; k<${p}; k++)
    {
        for (int v=0; v<${nvars}; v++)
        {
            //*((${dtype}${vec_width} *) &b_shr[SSOA_IDX(el0, v) + (os + k*${p*p})*${nvars*vec_width}]) = *(${dtype}${vec_width}*)(static_cast <${dtype} *>(__builtin_assume_aligned(b_glb + SOA_IDX(e0, v) + (os + k*${p*p})*ldb, 64)));

            *((${dtype}${vec_width} *) &b_glb[SOA_IDX(e0, v) + (os + k*${p*p})*ldb]) = *(${dtype}${vec_width}*)(b_shr + SSOA_IDX(el0, v) + (os + k*${p*p})*${nvars*vec_width});
        }
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

    extern __shared__ ${dtype} bs[];

    // Store Global to shared Shared Memory
    if(threadIdx.z < ${int(block_elem/vec_width)})
        ${funcn}_import(b, ldb, bs);
    __syncthreads();

    if(eg < n)
    {    
        ${funcn}_line(eg, el, bs, ${nvars*vec_width}, bs + ${nvars*block_elem*p*p*p}, ${nvars*vec_width});
    }
    if(threadIdx.z < ${int(block_elem/vec_width)})
        ${funcn}_export(bs + ${nvars*block_elem*p*p*p}, c, ldc);
}