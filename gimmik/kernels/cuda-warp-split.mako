# -*- coding: utf-8 -*-

<%namespace module='gimmik.generate.split' name='s'/>
<%namespace module='gimmik.utils' name='gimmik'/>

__global__ __launch_bounds__(${block_dim}) void
${funcn}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
    int i = ${rep*32}*blockIdx.x + (threadIdx.x % 32) + 32*(threadIdx.x/${32*split});
    int warp = (threadIdx.x / 32) % ${split};
    int osb = ((threadIdx.x % 32) + 32*(threadIdx.x/${32*split}))*${gimmik.ncols(mat)};
    ${dtype} dotp;
    //${dtype} extern __shared__ bs[];
    ${dtype} __shared__ bs[${int(block_dim/split)*gimmik.ncols(mat)}];

    if (i < n)
    {
        ${s.generator(mat=mat, beta=beta, shr_name='bs', shr_os='osb', shr_size=96000, split=split, rep=rep)}
    }
}
