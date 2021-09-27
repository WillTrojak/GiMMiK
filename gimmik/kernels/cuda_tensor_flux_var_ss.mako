# -*- coding: utf-8 -*-

/* CHANGE LOG
    Functions split off to reduce register pressure
*/ 

#include <cooperative_groups.h>
//#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
//#include <cuda/pipeline>

#define SOA_IDX(i, v) ((((i) / ${soasz})*${nvars} + (v))*${soasz} + (i) % ${soasz})
#define SSOA_IDX(i, v) ((((i) / ${vec_width})*${nvars} + (v))*${vec_width} + (i) % ${vec_width})
#define LCL_E(t) ( ${vec_width}*((t) / ${vec_width*nvars}) + ((t) % ${vec_width}))
#define ABS_E(t, b) ((b)*(${block_elem}) + LCL_E(t))
namespace cg = cooperative_groups;

__constant__ ${dtype} d[${p*p}] = {${','.join(str(D[j,i]) for j in range(p) for i in range(p))} };

__device__ ${dtype}
${funcn}_flux_shr(int v, int el, 
                  const ${dtype}* const __restrict__ u, int osu, 
                  ${dtype} const jac0,
                  ${dtype} const jac1,
                  ${dtype} const jac2
                 )
{
    __builtin_assume(v < ${nvars});

    ${dtype} a;
    if (v == 0)
    {
        a = ${c['zeta']}*(jac0*u[SSOA_IDX(el, 1) + osu] + jac1*u[SSOA_IDX(el, 2) + osu] + jac2*u[SSOA_IDX(el, 3) + osu]);
    }
    else if (v < 4)
    {
        ${dtype} fx = u[SSOA_IDX(el, 1) + osu]*u[SSOA_IDX(el, v) + osu] - ${c['nu']}*u[SSOA_IDX(el, 1 + v*3) + osu];
        ${dtype} fy = u[SSOA_IDX(el, 2) + osu]*u[SSOA_IDX(el, v) + osu] - ${c['nu']}*u[SSOA_IDX(el, 2 + v*3) + osu];
        ${dtype} fz = u[SSOA_IDX(el, 3) + osu]*u[SSOA_IDX(el, v) + osu] - ${c['nu']}*u[SSOA_IDX(el, 3 + v*3) + osu];
        a = jac0*fx + jac1*fy + jac2*fz + (max(2 - v,0)*jac0 + (v - 1)*(3 - v)*jac1 + max(v - 2, 0)*jac2)*u[SSOA_IDX(el, 0) + osu];
    }
    else
    {
        int idx = (v - 4)/3;
        int i = v - 4 - 3*idx;
        a = -${1/c['tr']}*(max(1 - i,0)*jac0 + i*(2 - i)*jac1 + max(i - 1, 0)*jac2)*u[idx + 1];
    }
    return a;
}

__device__ ${dtype}
${funcn}_flux_glb(int v, int eg,
                  const ${dtype}* const __restrict__ u, int osu,
                  ${dtype} const jac0,
                  ${dtype} const jac1,
                  ${dtype} const jac2
                 )
{
    __builtin_assume(v < ${nvars});

    ${dtype} a;
    if (v == 0)
    {
        a = ${c['zeta']}*(jac0*u[SOA_IDX(eg, 1) + osu] + jac1*u[SOA_IDX(eg, 2) + osu] + jac2*u[SOA_IDX(eg, 3) + osu]);
    }
    else if (v < 4)
    {
        ${dtype} fx = u[SOA_IDX(eg, 1) + osu]*u[SOA_IDX(eg, v) + osu] - ${c['nu']}*u[SOA_IDX(eg, 1 + v*3) + osu];
        ${dtype} fy = u[SOA_IDX(eg, 2) + osu]*u[SOA_IDX(eg, v) + osu] - ${c['nu']}*u[SOA_IDX(eg, 2 + v*3) + osu];
        ${dtype} fz = u[SOA_IDX(eg, 3) + osu]*u[SOA_IDX(eg, v) + osu] - ${c['nu']}*u[SOA_IDX(eg, 3 + v*3) + osu];
        a = jac0*fx + jac1*fy + jac2*fz + (max(2 - v,0)*jac0 + (v - 1)*(3 - v)*jac1 + max(v - 2, 0)*jac2)*u[SOA_IDX(eg, 0) + osu];
    }
    else
    {
        int idx = (v - 4)/3;
        int i = v - 4 - 3*idx;
        a = -${1/c['tr']}*(max(1 - i,0)*jac0 + i*(2 - i)*jac1 + max(i - 1, 0)*jac2)*u[SOA_IDX(eg, idx + 1) + osu];
    }
    return a;
}

__device__ ${dtype}
${funcn}_shared(int v, int el, int j,
                const ${dtype}* const __restrict__ b, int osb, int ldb,
                ${dtype} const jac0,
                ${dtype} const jac1,
                ${dtype} const jac2
               )
{
    ${dtype} dotp = 0.;
    for (int i=0; i<${p}; i++)
        dotp += d[i*${p} + j]*${funcn}_flux_shr(v, el, b, (osb + i*ldb)*${nvars*vec_width}, jac0, jac1, jac2);
    return dotp;
}

__device__ ${dtype}
${funcn}_global(int v, int eg, int j, int osb, int ldb_l,
                const ${dtype}* const __restrict__ b, int ldb,
                ${dtype} const jac0,
                ${dtype} const jac1,
                ${dtype} const jac2
               )
{

    ${dtype} dotp = 0.;
    for(int i=0; i<${p}; i++)
        dotp += d[i*${p} + j]*${funcn}_flux_glb(v, eg, b, (osb + i*ldb_l)*ldb, jac0, jac1, jac2);
    return dotp;
}

__device__ void
${funcn}_store(cg::thread_block_tile<${int(block_elem/vec_width)}> tile, 
               const ${dtype}* const __restrict__ b,
               ${dtype}* __restrict__ a, int osa, int lda
               )
{

    int el0 = ${vec_width}*tile.thread_rank();
    int e0 = blockIdx.x*${block_elem} + ${vec_width}*tile.thread_rank();
    
    for (int j=0; j<${p*p}; j++)
    {
        for (int v=0; v<${nvars}; v++)
        {
            *((${dtype}${vec_width} *) &a[SOA_IDX(e0, v) + (osa + j*${p})*lda]) = *(${dtype}${vec_width}*)(b + SSOA_IDX(el0, v) + j*${nvars*vec_width});
        }
    }

    return;
}

__device__ void
${funcn}_load(cg::thread_block_tile<${int(block_elem/vec_width)}> tile, 
              const ${dtype}* const __restrict__ a, int osa, int lda,
              ${dtype}* __restrict__ b)
{

    int el0 = ${vec_width}*tile.thread_rank();
    int e0  = blockIdx.x*${block_elem} + ${vec_width}*tile.thread_rank();

    for (int j=0; j<${p*p}; j++)
    {
        for (int v=0; v<${nvars}; v++)
        {
            *((${dtype}${vec_width} *) &b[SSOA_IDX(el0, v) + j*${nvars*vec_width}]) = *(${dtype}${vec_width}*)(a + SOA_IDX(e0, v) + (osa + j*${p})*lda);
        }
    }
    return;
}

__constant__ ${dtype} jacx[] = {1., 0., 0.};
__constant__ ${dtype} jacy[] = {0., 1., 0.};
__constant__ ${dtype} jacz[] = {0., 0., 1.};

__global__ void
__launch_bounds__ (${blk_dim})
${funcn}(int n,
         const ${dtype}* const __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<${int(block_elem/vec_width)}>(block);
    
    // Element No. in global frame
    int el = threadIdx.x/${vec_width*nvars} + threadIdx.x % ${vec_width};
    int eg = (el < ${block_elem}) ? ABS_E(threadIdx.x, blockIdx.x) : n;

    // z-plane evaluated by thread 
    int var = (threadIdx.x/${vec_width}) % ${nvars};

    extern __shared__ ${dtype} bs[];
    ${dtype} dotp0, dotp1, dotp2;

% for i in range(p): # z-y plane loop
    if(tile.meta_group_rank() == 0)
        ${funcn}_load(tile, b, ${i}, ldb, bs);
    __syncthreads();

    if(eg < n)
    {
    % for k in range(p):
    % for j in range(p):
        dotp0 = ${funcn}_global(var, eg, ${i}, ${j*p + k*p*p}, 1, b, ldb, jacx[0], jacx[1], jacx[2]); // x-line
        dotp1 = ${funcn}_shared(var, el, ${j}, bs, ${k*p}, 1, jacy[0], jacy[1], jacy[2]); // y-line
        dotp2 = ${funcn}_shared(var, el, ${k}, bs, ${j}, ${p},  jacz[0], jacz[1], jacz[2]); // z-line
        //bs[${block_elem*nvars*p*p} + SSOA_IDX(el, var) + ${(j + k*p)*nvars*vec_width}] = dotp1 + dotp2;
        c[SOA_IDX(eg, var) + ${i + j*p + k*p*p}*ldc] = dotp0 + dotp1 + dotp2;
    % endfor
    % endfor
    }
    
    __syncthreads();
    //if(tile.meta_group_rank() == tile.meta_group_size() -1)
    //   ${funcn}_store(tile, bs + ${block_elem*nvars*p*p}, c, ${i}, ldc);
% endfor
}
