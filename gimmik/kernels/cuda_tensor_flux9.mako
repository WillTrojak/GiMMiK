# -*- coding: utf-8 -*-
<%namespace module='gimmik.flux2' name='flux'/>
<%namespace module='gimmik.utils' name='utils'/>

/* CHANGE LOG
    Same as flux8 but with:
    * explicit for loop for y-z plane
*/ 

#define AOSOA_V(i) ((((i) - ((i) % ${soasz}))) / ${soasz}) % ${nvars}
#define SOA_IDX(i, v) ((((i) / ${soasz})*${nvars} + (v))*${soasz} + (i) % ${soasz})
#define TSOA2_IDX(j, yo, eo) ( ${nvars}*((j) + (yo)) + (eo))

__global__ void
${funcn}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{

    // Memory common 
    int eg = (blockIdx.x*blockDim.x + threadIdx.x)/${p}; // element No. in global frame
    int el_offset = ${p*p*nvars}*(threadIdx.x/${p}); // element No. in local frame times offset (simplify div. at your peril)
    ${dtype} ze = ${0.};

    __shared__ ${dtype} bs[${p*nvars*blk_dim}];

    ${dtype} l_acc[${nvars}], bl[${nvars*p}], blx[${nvars}];

    // Load differentiation chars into memory
    ${dtype} d[${p*p}];
% for i in range(p):
% for j in range(p):
    d[${i*p + j}] = ${D[j,i]};
% endfor
% endfor

    // n is now number of elements
    if (eg < n)
    {

        int thrd_k = threadIdx.x % ${p};  // z-plane evaluated by thread
        int kz_offset = ${p*p}*thrd_k;  // z-plane offset
        int ky_offset = ${p}*thrd_k;  // initial y-plane offset 

        for (int i_s=0; i_s<${p}; i_s++){ // z-y plane loop

            // Read z-y plane @ x=i_s to shared
            for (int k=0; k<${p}; k++){
                for (int v=0; v<${nvars}; v++){
                    bs[v + TSOA2_IDX(k, ky_offset, el_offset)] = b[SOA_IDX(eg, v) + (i_s + ky_offset + k*${p*p})*ldb];
                }
            }
            __syncthreads();

            // Explicitly put y-line in register
        % for j in range(p):
        % for v in range(nvars): # var loop 2
            bl[${v + j*nvars}] = bs[${v} + TSOA2_IDX(thrd_k, ${j*p}, el_offset)]; 
        % endfor # end var loop 2
        % endfor

        % for j in range(p): # y-line loop

            // x-line contribution
        % for i in range(p): # x-line loop
            //   Explicitly put x point in register
        % for v in range(nvars): # var loop 3
            blx[${v}] = ${utils.glb_addr(name='b', i='eg', v=v, o=f'({i + j*p} + kz_offset)*ldb')};
        % endfor # var loop 3
        % for v in range(nvars): # var loop 4
            l_acc[${v}] ${('+','')[i==0]}= ${'d[i_s*{p}+{j}]*{f}'.format(p=p, j=j,
                                             f=flux.src(func='linadv_n', name='blx', jac=['1','ze','ze'],
                                             ndims=ndims, idx=[f'{v}']))};
        % endfor # end var loop 4
        % endfor # end x-line loop

        % for v in range(nvars): # var loop 5
            l_acc[${v}] += ${' + '.join('{d}*{f}'.format(d=D[i,j], 
                             f=flux.src(func='linadv_n', name='bl', jac=['ze','1','ze'], ndims=ndims, 
                                         sep='+', idx=[f'{v}', f'{i*nvars}'])) for i in range(p))} +
                           ${' + '.join('d[ky_offset+{i}]*{f}'.format(i=i, 
                             f=flux.src(func='linadv_n', name='bs', jac=['ze','ze','1'], ndims=ndims, 
                                        sep='+', idx=[f'{v}', f'TSOA2_IDX({i}, {j*p}, el_offset)'])) for i in range(p))};
            c[SOA_IDX(eg, ${v}) + (i_s + ${j*p} + kz_offset)*ldc] = l_acc[${v}];
        % endfor # end var loop 5

        % endfor # end y-line loop

        }

    }
}

#undef AOSOA_V
#undef SOA_IDX
#undef TSOA2_IDX