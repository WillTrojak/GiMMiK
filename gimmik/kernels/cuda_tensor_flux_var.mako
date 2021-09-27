# -*- coding: utf-8 -*-

/* CHANGE LOG
    Functions split off to reduce register pressure
*/ 

#include <cooperative_groups.h>
//#include <cooperative_groups/memcpy_async.h>
//#include <cuda/barrier>
//#include <cuda/pipeline>

#define SOA_IDX(i, v) ((((i) / ${soasz})*${nvars} + (v))*${soasz} + (i) % ${soasz})
#define ABS_E(t, b) ((b)*(${block_elem}) + (t)/${nvars})

namespace cg = cooperative_groups;

__constant__ ${dtype} d[${p*p}] = {${','.join(str(D[j,i]) for j in range(p) for i in range(p))} };

__device__ ${dtype}
${funcn}_flux_shr(int v, 
                  const ${dtype}* const __restrict__ u,
                  ${dtype} const jac0,
                  ${dtype} const jac1,
                  ${dtype} const jac2
                 )
{
    __builtin_assume(v < ${nvars});

    ${dtype} a;
    if (v == 0)
    {
        a = ${c['zeta']}*(jac0*u[1] + jac1*u[2] + jac2*u[3]);
    }
    else if (v < 4)
    {
        ${dtype} fx = u[1]*u[v] - ${c['nu']}*u[1 + v*3];
        ${dtype} fy = u[2]*u[v] - ${c['nu']}*u[2 + v*3];
        ${dtype} fz = u[3]*u[v] - ${c['nu']}*u[3 + v*3];
        a = jac0*fx + jac1*fy + jac2*fz + (max(2 - v,0)*jac0 + (v - 1)*(3 - v)*jac1 + max(v - 2, 0)*jac2)*u[0];
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
${funcn}_flux_glb(int v, int eg, int i,
                  const ${dtype}* const __restrict__ u, int ldu,
                  ${dtype} const jac0,
                  ${dtype} const jac1,
                  ${dtype} const jac2
                 )
{
    __builtin_assume(v < ${nvars});

    ${dtype} a;
    if (v == 0)
    {
        a = ${c['zeta']}*(jac0*u[SOA_IDX(eg, 1) + v*ldu] + jac1*u[SOA_IDX(eg, 2) + v*ldu] + jac2*u[SOA_IDX(eg, 3) + v*ldu]);
    }
    else if (v < 4)
    {
        ${dtype} fx = u[SOA_IDX(eg, 1) + i*ldu]*u[SOA_IDX(eg, v) + i*ldu] - ${c['nu']}*u[SOA_IDX(eg, 1 + v*3) + i*ldu];
        ${dtype} fy = u[SOA_IDX(eg, 2) + i*ldu]*u[SOA_IDX(eg, v) + i*ldu] - ${c['nu']}*u[SOA_IDX(eg, 2 + v*3) + i*ldu];
        ${dtype} fz = u[SOA_IDX(eg, 3) + i*ldu]*u[SOA_IDX(eg, v) + i*ldu] - ${c['nu']}*u[SOA_IDX(eg, 3 + v*3) + i*ldu];
        a = jac0*fx + jac1*fy + jac2*fz + (max(2 - v,0)*jac0 + (v - 1)*(3 - v)*jac1 + max(v - 2, 0)*jac2)*u[SOA_IDX(eg, 0) + i*ldu];
    }
    else
    {
        int idx = (v - 4)/3;
        int i = v - 4 - 3*idx;
        a = -${1/c['tr']}*(max(1 - i,0)*jac0 + i*(2 - i)*jac1 + max(i - 1, 0)*jac2)*u[SOA_IDX(eg, idx + 1) + i*ldu];
    }
    return a;
}

__device__ ${dtype}
${funcn}_shared(int v, int j,
                const ${dtype}* const __restrict__ b, int osb, int ldb,
                ${dtype} const jac0,
                ${dtype} const jac1,
                ${dtype} const jac2
               )
{
    ${dtype} dotp = 0.;
    for (int i=0; i<${p}; i++)
        dotp += d[i*${p} + j]*${funcn}_flux_shr(v, b + osb + i*ldb, jac0, jac1, jac2);
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
        dotp += d[i*${p} + j]*${funcn}_flux_glb(v, eg, osb + i*ldb_l, b, ldb, jac0, jac1, jac2);
    return dotp;
}

__device__ void
${funcn}_store(cg::thread_block_tile<${int(block_elem/vec_width)}> tile, 
               const ${dtype}* const __restrict__ b,
               ${dtype}* __restrict__ a, int osa, int lda
               )
{

    int e0 = blockIdx.x*${block_elem} + ${vec_width}*tile.thread_rank();
    
% if vec_width == 4:
    int4 t_enum = make_int4((4*tile.thread_rank() + 0)*${p*p*nvars}, 
                            (4*tile.thread_rank() + 1)*${p*p*nvars}, 
                            (4*tile.thread_rank() + 2)*${p*p*nvars}, 
                            (4*tile.thread_rank() + 3)*${p*p*nvars}
                           );

    for (int j=0; j<${p*p}; j++)
    {
#pragma unroll
        for (int v=0; v<${nvars}; v++)
        {
            *((${dtype}4 *) &a[SOA_IDX(e0, v) + (osa + j*${p})*lda]) = 
                make_${dtype}4(b[t_enum.x + v + j*${nvars}],
                               b[t_enum.y + v + j*${nvars}],
                               b[t_enum.z + v + j*${nvars}],
                               b[t_enum.w + v + j*${nvars}]
                              );
        }
    }
% elif vec_width == 2:
    int2 t_enum = make_int2((2*tile.thread_rank() + 0)*${p*p*nvars}, 
                            (2*tile.thread_rank() + 1)*${p*p*nvars}
                           );

    for (int j=0; j<${p*p}; j++)
    {
#pragma unroll
        for (int v=0; v<${nvars}; v++)
        {
            *((${dtype}2 *) &a[SOA_IDX(e0, v) + (osa + j*${p})*lda]) = 
                make_${dtype}2(b[t_enum.x + v + j*${nvars}],
                               b[t_enum.y + v + j*${nvars}]
                              );
        }
    }
% endif

    return;
}


//__device__ void
//${funcn}_load_v(int n, int eg, )
//{
//    for (int j=0; j<${p*p},)
//    
//}

__device__ void
${funcn}_load(cg::thread_block_tile<${int(block_elem/vec_width)}> tile, 
              const ${dtype}* const __restrict__ a, int osa, int lda,
              ${dtype}* __restrict__ b)
{

    int e0 = blockIdx.x*${block_elem} + ${vec_width}*tile.thread_rank();
    
% if vec_width == 4:
    int4 t_enum = make_int4((4*tile.thread_rank() + 0)*${p*p*nvars}, 
                            (4*tile.thread_rank() + 1)*${p*p*nvars}, 
                            (4*tile.thread_rank() + 2)*${p*p*nvars}, 
                            (4*tile.thread_rank() + 3)*${p*p*nvars}
                           );
    for (int j=0; j<${p*p}; j++)
    {
        for (int v=0; v<${nvars}; v++)
        {
            ${dtype}4 temp = *(${dtype}4*)(a + SOA_IDX(e0, v) + (osa + j*${p})*lda);
            b[t_enum.x + v + j*${nvars}] = temp.x;
            b[t_enum.y + v + j*${nvars}] = temp.y;
            b[t_enum.z + v + j*${nvars}] = temp.z;
            b[t_enum.w + v + j*${nvars}] = temp.w;
        }
    }
% elif vec_width == 2:
    int2 t_enum = make_int2((2*tile.thread_rank() + 0)*${p*p*nvars}, 
                            (2*tile.thread_rank() + 1)*${p*p*nvars}
                           );
    for (int j=0; j<${p*p}; j++)
    {
        for (int v=0; v<${nvars}; v++)
        {
            ${dtype}2 temp = *(${dtype}2*)(a + SOA_IDX(e0, v) + (osa + j*${p})*lda);
            b[t_enum.x + v + j*${nvars}] = temp.x;
            b[t_enum.y + v + j*${nvars}] = temp.y;
        }
    }
% endif
    
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
    int eg = (threadIdx.x / ${nvars} < ${int(block_elem)}) ? ABS_E(threadIdx.x, blockIdx.x) : n;
 
    // Shared Memory Offset
    int el_offset = ${nvars*p*p}*(threadIdx.x / ${nvars}); //Currently no bank deconfliction
    
    // z-plane evaluated by thread 
    int var = threadIdx.x % ${nvars};

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
        dotp1 = ${funcn}_shared(var, ${j}, bs + el_offset, ${k*p*nvars}, 1, jacy[0], jacy[1], jacy[2]); // y-line
        dotp2 = ${funcn}_shared(var, ${k}, bs + el_offset, ${j*nvars}, ${p},  jacz[0], jacz[1], jacz[2]); // z-line
        bs[${block_elem*nvars*p*p} + el_offset + var + ${(j + k*p)*nvars}] = dotp0 + dotp1 + dotp2;
        //c[SOA_IDX(eg, var) + ${i + j*p + k*p*p}*ldc] = dotp0 + dotp1 + dotp2;
    % endfor
    % endfor
    }
    
    __syncthreads();
    if(tile.meta_group_rank() == tile.meta_group_size() -1)
       ${funcn}_store(tile, bs + ${block_elem*nvars*p*p}, c, ${i}, ldc);
% endfor
}
