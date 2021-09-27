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
${funcn}_line(int v, int j,
            % for v in range(nvars):
                ${dtype}* __restrict__ b_${v},
            % endfor 
                int ldb,
                ${dtype} const jac0,
                ${dtype} const jac1,
                ${dtype} const jac2
               )
{

    ${dtype} dotp = 0.;
    for(int osb=0; osb<${p}*ldb; osb+=ldb)
    {
        ${dtype} a;
        if (v == 0)
        {
            a = ${c['zeta']}*(jac0*b_1[osb] + jac1*b_2[osb] + jac2*b_3[osb]);
        }
        else if (v < 4)
        {
            ${dtype} s0 = max(2 - v, 0), s1 = (v - 1)*(3 - v), s2 = max(v - 2, 0);
            ${dtype} b_v = s0*b_1[osb] + s1*b_2[osb] + s2*b_3[osb];

            a  = jac0*(b_1[osb]*b_v - ${c['nu']}*(s0*b_4[osb] + s1*b_7[osb] + s2*b_10[osb]));
            a += jac1*(b_2[osb]*b_v - ${c['nu']}*(s0*b_5[osb] + s1*b_8[osb] + s2*b_11[osb]));
            a += jac2*(b_3[osb]*b_v - ${c['nu']}*(s0*b_6[osb] + s1*b_9[osb] + s2*b_12[osb]));
            a += (s0*jac0 + s1*jac1 + s2*jac2)*b_0[osb];
        }
        else
        {
            int idx = (v - 4)/3;
            int i = v - 4 - 3*idx;
            a = max(1 - idx, 0)*b_1[osb] + idx*(2 - idx)*b_2[osb] + max(idx - 1, 0)*b_3[osb];
            a *= -${1/c['tr']}*(max(1 - i,0)*jac0 + i*(2 - i)*jac1 + max(i - 1, 0)*jac2);
        }
        dotp += a;
    }
    return dotp;
}

__device__ void
${funcn}_store(cg::thread_block_tile<${int(block_elem/vec_width)}> tile, 
               ${dtype}* __restrict__ b,
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
              ${dtype}* __restrict__ a, int osa, int lda,
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

__global__ void
__launch_bounds__ (${blk_dim})
${funcn}(int n,
         ${dtype}* __restrict__ b, int ldb,
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
    ${dtype} jacx[] = {1., 0., 0.}, jacy[] = {0., 1., 0.}, jacz[] = {0., 0., 1.};

% for i in range(p): # z-y plane loop
    if(tile.meta_group_rank() == 0)
        ${funcn}_load(tile, b, ${i}, ldb, bs);
    __syncthreads();

    if(eg < n)
    {
    % for k in range(p):
    % for j in range(p):
        dotp0 = ${funcn}_line(var, ${i},
        % for v in range(nvars):
            b + SOA_IDX(eg, ${v}) + ${j*p + k*p*p}*ldb,
        % endfor
            1*ldb, jacx[0], jacx[1], jacx[2]); // x-line
        dotp1 = ${funcn}_line(var, ${j},
        % for v in range(nvars):
            bs + SSOA_IDX(el, ${v}) + ${k*p*nvars*vec_width},
        % endfor
            ${nvars*vec_width}, jacy[0], jacy[1], jacy[2]); // y-line
        dotp2 = ${funcn}_line(var, ${k},
        % for v in range(nvars):
            bs + SSOA_IDX(el, ${v}) + ${j*nvars*vec_width},
        % endfor
            ${p*nvars*vec_width}, jacz[0], jacz[1], jacz[2]); // z-line

        bs[${block_elem*nvars*p*p} + SSOA_IDX(el, var) + ${(j + k*p)*nvars*vec_width}] = dotp0 + dotp1 + dotp2;
        //c[SOA_IDX(eg, var) + ${i + j*p + k*p*p}*ldc] = dotp0 + dotp1 + dotp2;
    % endfor
    % endfor
    }
    
    __syncthreads();
    if(tile.meta_group_rank() == tile.meta_group_size() -1)
       ${funcn}_store(tile, bs + ${block_elem*nvars*p*p}, c, ${i}, ldc);
% endfor
}
