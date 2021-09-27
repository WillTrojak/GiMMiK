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
    BLOCK CONFIG       = (${block_elem*nvars}, ${p}, ${p})
*/


#include <cooperative_groups.h>
//#include <cooperative_groups/memcpy_async.h>
//#include <cuda/barrier>
//#include <cuda/pipeline>

#define SOA_IDX(i, v) ((((i) / ${soasz})*${nvars} + (v))*${soasz} + (i) % ${soasz})
#define SSOA_IDX(i, v) ((((i) / ${vec_width})*${nvars} + (v))*${vec_width} + (i) % ${vec_width})

namespace cg = cooperative_groups;

__constant__ ${dtype} d[${p*p}] = {${','.join(str(D[j,i]) for j in range(p) for i in range(p))} };

__device__ ${dtype}
${funcn}_flux(int v, 
            % for v in range(nvars):
                ${dtype}* __restrict__ u_${v},
            % endfor
                ${dtype} const jac0,
                ${dtype} const jac1,
                ${dtype} const jac2
             )
{

    if (v == 0)
    {
        return ${c['zeta']}*(jac0*u_1[0] + jac1*u_2[0] + jac2*u_3[0]);
    }
    else if (v < 4)
    {
        ${dtype} s0 = max(2 - v, 0), s1 = (v - 1)*(3 - v), s2 = max(v - 2, 0);
        ${dtype} u_v = s0*u_1[0] + s1*u_2[0] + s2*u_3[0];

        ${dtype} a  = jac0*(u_1[0]*u_v - ${c['nu']}*(s0*u_4[0] + s1*u_7[0] + s2*u_10[0]));
        a += jac1*(u_2[0]*u_v - ${c['nu']}*(s0*u_5[0] + s1*u_8[0] + s2*u_11[0]));
        a += jac2*(u_3[0]*u_v - ${c['nu']}*(s0*u_6[0] + s1*u_9[0] + s2*u_12[0]));
        a += (s0*jac0 + s1*jac1 + s2*jac2)*u_0[0];
        return a;
    }
    else
    {
        int idx = (v - 4)/3;
        int i = v - 4 - 3*idx;
        ${dtype} a = max(1 - idx, 0)*u_1[0] + idx*(2 - idx)*u_2[0] + max(idx - 1, 0)*u_3[0];
        a *= -${1/c['tr']}*(max(1 - i,0)*jac0 + i*(2 - i)*jac1 + max(i - 1, 0)*jac2);
        return a;
    }
    return 0.;
}

__device__ void
${funcn}_line(int eg, int el,
              ${dtype}* __restrict__ b, int ldb,
              ${dtype}* __restrict__ c, int ldc
             )
{
    __builtin_assume(ldb > 0);
    __builtin_assume(ldc > 0);

    int i  = threadIdx.y;
    int j  = threadIdx.z;
    int v  = (threadIdx.x / ${block_elem}) % ${nvars};

    ${dtype} jacx[] = {1., 0., 0.}, jacy[] = {0., 1., 0.}, jacz[] = {0., 0., 1.};

    for(int k=0; k<${p}; k++)
    {
        ${dtype} dotp_x = 0.;
        ${dtype} dotp_y = 0.;
        ${dtype} dotp_z = 0.;
        for(int x=0; x<${p}; x++)
        {
            dotp_x += d[x*${p} + j]*${funcn}_flux(v, 
            % for v in range(nvars):
                b + SSOA_IDX(el, ${v}) + (x + j*${p} + k*${p*p})*ldb,
            % endfor
                jacx[0], jacx[1], jacx[2]);

            dotp_y += d[x*${p} + i]*${funcn}_flux(v, 
            % for v in range(nvars):
                b + SSOA_IDX(el, ${v}) + (i + x*${p} + k*${p*p})*ldb,
            % endfor
                jacy[0], jacy[1], jacy[2]);

            dotp_z += d[x*${p} + k]*${funcn}_flux(v, 
            % for v in range(nvars):
                b + SSOA_IDX(el, ${v}) + (i + j*${p} + x*${p*p})*ldb,
            % endfor
                jacz[0], jacz[1], jacz[2]);
        }
        c[SOA_IDX(eg, v) + (i + j*${p} + k*${p*p})*ldc] = dotp_x + dotp_y + dotp_z;
    }
}

__device__ void
${funcn}_import(${dtype}* __restrict__ b_glb, int ldb,
                ${dtype}* __restrict__ b_shr)
{
    __builtin_assume(ldb > 0);

    int el0 = threadIdx.x*${vec_width};
    int e0  = blockIdx.x*${block_elem} + el0;

    int os = threadIdx.y + threadIdx.z*${p};

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

__global__ void
__launch_bounds__ (${block_elem*nvars*p*p})
${funcn}(int n,
         ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
    auto block = cg::this_thread_block();
    
    // Element No. in global frame
    int el = threadIdx.x % ${block_elem};
    int eg = (el < ${block_elem}) ? (blockIdx.x*${block_elem} + el) : n;

    extern __shared__ ${dtype} bs[];

    // Store Global to shared Shared Memory
    if(blockIdx.x*${block_elem} + threadIdx.x < n & threadIdx.x < ${int(block_elem/vec_width)})
        ${funcn}_import(b, ldb, bs + threadIdx.x*${vec_width*nvars*p*p*p});
    __syncthreads();

    if(eg < n)
    {    
        ${funcn}_line(eg, el, bs, ${nvars*vec_width}, c, ldc);
    }
}
