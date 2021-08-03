# -*- coding: utf-8 -*-

__global__
% if block_dim is not None:
__launch_bounds__(${block_dim})
% endif
void
${funcn}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
    int i = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
    ${dtype} dotp;

    if (i < n)
    {
    % for j, jx in enumerate(mat):
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
    }
}
