# -*- coding: utf-8 -*-
<%namespace module='gimmik.flux2' name='flux'/>
<%namespace module='gimmik.utils' name='utils'/>

/* CHANGE LOG
    Same as flux11 but with:
    * shared bank deconflicting
    * d in constant memory
*/

//#include <stdio.h>

#define BLOCK_DIMX ${bcfg.blk_dim}
#define SOA_SZ ${soasz}
#define SHR_SIZE ${bcfg.shr_size}
#define ORDER ${p-1}
#define WTYPE ${dtype}
#define WARP_SIZE ${bcfg.warp_size}
#define ELEM_PER_WARP ${bcfg.elem_warp_max}

#define SOA_IDX(i, v) ((((i) / ${soasz})*${nvars} + (v))*${soasz} + (i) % ${soasz})
#define TSOA2_IDX(j, yo, eo) ( ${nvars}*((j) + (yo)) + (eo))

#define ELEM_K(t) (((t) % ${bcfg.warp_size}) % ${p})
#define WARP(t) ((t)/${bcfg.warp_size})
#define ATHRD (${bcfg.warp_size} - ${bcfg.warp_size} % ${p})
#define ABS_E(t, b, bx) ((b)*(ATHRD / ${p})*(bx / 32) + WARP(t)*(ATHRD / ${p}) + ((t) % ${bcfg.warp_size}) / ${p})
#define LCL_E(t) (WARP(t)*(ATHRD / ${p}) + ((t) % ${bcfg.warp_size}) / ${p})
#define WRP_E(t) (((t) % ${bcfg.warp_size} ) / ${p})
#define W_MASK(t) (((1 << ${p}) - 1) << (${p}*(((t) % ${bcfg.warp_size}) / ${p})))

__constant__ ${dtype} d[${p*p}] = {${','.join(str(D[j,i]) for j in range(p) for i in range(p))} };

__global__ void
__launch_bounds__ (${bcfg.blk_dim})
${funcn}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{

    // Element No. in global frame
    int eg = ((threadIdx.x % ${bcfg.warp_size}) < ATHRD) ? ABS_E(threadIdx.x, blockIdx.x, blockDim.x) : n;
    // Shared Memory Offset
    int el_offset = ${bcfg.shr_size_elem}*LCL_E(threadIdx.x) + ((${bcfg.warp_size} - ((${bcfg.shr_offset_rem}*WRP_E(threadIdx.x))%${bcfg.warp_size})) % ${bcfg.warp_size} + WRP_E(threadIdx.x)*${p}) % ${bcfg.warp_size};
    extern __shared__ ${dtype} bs[]; // shared array size [elements] = ${bcfg.shr_var_size}

    // z-plane evaluated by thread 
    int thrd_k = ELEM_K(threadIdx.x);  

    ${dtype} ze = ${0.}, on = ${1.};

    ${dtype} acl[${nvars}], bl[${nvars*p}], blx[${nvars}];

    // n is now number of elements
    if (eg < n)
    {
    % for i_s in range(p): # z-y plane loop

    % if i_s > 0:
        __syncthreads();
    % endif
        // Read y-line into register and use that to build z-y plane @ x=${i_s} to shared
    % for j in range(p):
    % for v in range(nvars): # var loop 1
        bl[${v + j*nvars}] = b[SOA_IDX(eg, ${v}) + (${i_s} + ${j*p} + ${p*p}*thrd_k)*ldb];
    % endfor # end var loop 1
    % endfor
    % for j in range(p):
    % for v in range(nvars): # var loop 1
        bs[${v} + TSOA2_IDX(thrd_k, ${j*p}, el_offset)] = bl[${v + j*nvars}];
    % endfor # end var loop 1
    % endfor

    % for j in range(p): # y-line loop

        // x-line contribution
    % for i in range(p): # x-line loop
        // Explicitly put x point in register

    % if i == i_s:
    % for v in range(nvars): # var loop 4
        acl[${v}] =${(f' acl[{v}] +','')[i==0]} ${'{d}*{f}'.format(d=D[i_s,i], 
                                         f=flux.src(func=flux_n, name='bl', jac=['on','ze','ze'],
                                         ndims=ndims, sep='+', idx=[f'{v}', f'{j*nvars}']))};
    % endfor # end var loop 4
    % else:
    % for v in range(nvars): # var loop 3
        blx[${v}] = ${utils.glb_addr(name='b', i='eg', v=v, o=f'({i + j*p} + thrd_k*{p*p})*ldb')};
    % endfor # var loop 3
    % for v in range(nvars): # var loop 4
        acl[${v}] =${(f' acl[{v}] +','')[i==0]} ${'{d}*{f}'.format(d=D[i_s,i], 
                                         f=flux.src(func=flux_n, name='blx', jac=['on','ze','ze'],
                                         ndims=ndims, idx=[f'{v}']))};
    % endfor # end var loop 4
    % endif
    % endfor # end x-line loop

    % if j == 0:
        __syncthreads();
    % endif

    % for v in range(nvars): # var loop 5
        acl[${v}] = acl[${v}] + ${' + '.join('{d}*{f}'.format(d=D[j,i], 
                         f=flux.src(func=flux_n, name='bl', jac=['ze','on','ze'], ndims=ndims, 
                                     sep='+', idx=[f'{v}', f'{i*nvars}'])) for i in range(p))} + ${' + '.join('d[thrd_k*{p}+{i}]*{f}'.format(i=i, p=p,
                         f=flux.src(func=flux_n, name='bs', jac=['ze','ze','on'], ndims=ndims, 
                                    sep='+', idx=[f'{v}', f'TSOA2_IDX({i}, {j*p}, el_offset)'])) for i in range(p))};
        c[SOA_IDX(eg, ${v}) + (${i_s + j*p} + thrd_k*${p*p})*ldc] = acl[${v}];
    % endfor # end var loop 5

    % endfor # end y-line loop

    % endfor # end z-y plane loop

    }
}

#undef TSOA2_IDX
#undef ELEM_K
#undef WARP
#undef ATHRD
#undef ABS_E
#undef LCL_E
#undef WRP_E
#undef W_MASK