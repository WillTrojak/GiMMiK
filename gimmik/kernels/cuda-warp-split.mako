# -*- coding: utf-8 -*-

__global__
__launch_bounds__(${block_dim})
void
${funcn}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
    int i = ${32}*blockIdx.x + (threadIdx.x % 32);
    int warp = threadIdx.x / 32;
    ${dtype} dotp;

    if (i < n)
    {
    % for w in range(split):
        if (warp == ${w})
        {
        % for j, jx in enumerate(mat):
        % if w*row_per_warp <= j < (w+1)*row_per_warp:
            dotp = ${' + '.join('{kx}*b[i + {k}*ldb]'.format(k=k, kx=kx)
                                for k, kx in enumerate(jx) if kx != 0) or 0};
        % if beta == 0:
            c[i + ${j}*ldc] = dotp;
        % elif beta == 1:
            c[i + ${j}*ldc] += dotp;
        % else:
            c[i + ${j}*ldc] = dotp + ${beta}*c[i + ${j}*ldc];
        % endif
        % endfor
        % endfor
        }
    % endfor
    }
}
