# -*- coding: utf-8 -*-
<%namespace module='gimmik.flux2' name='flux'/>
<%namespace module='gimmik.utils' name='utils'/>

/* CHANGE LOG
    Functions split off to reduce register pressure
*/ 

#define SOA_IDX(i, v) ((((i) / ${soasz})*${nvars} + (v))*${soasz} + (i) % ${soasz})
#define TSOA2_IDX(j, yo, eo) ( ${nvars}*((j) + (yo)) + (eo))
#define GLOBAL_IDX(v, eg, i, ld) (SOA_IDX(i, v) + (i)*(ld))

#define ELEM_K(t) (((t) % ${bcfg.warp_size}) % ${p})
#define WARP(t) ((t)/${bcfg.warp_size})
#define ATHRD (${bcfg.warp_size} - ${bcfg.warp_size} % ${p})
#define ABS_E(t, b, bx) ((b)*(ATHRD / ${p})*(bx / 32) + WARP(t)*(ATHRD / ${p}) + ((t) % ${bcfg.warp_size}) / ${p})
#define LCL_E(t) (WARP(t)*(ATHRD / ${p}) + ((t) % ${bcfg.warp_size}) / ${p})
#define WRP_E(t) (((t) % ${bcfg.warp_size} ) / ${p})
#define W_MASK(t) (((1 << ${p}) - 1) << (${p}*(((t) % ${bcfg.warp_size}) / ${p})))

#define SHR_T true
#define SHR_N false

__constant__ ${dtype} d[${p*p}] = {${','.join(str(D[j,i]) for j in range(p) for i in range(p))} };

template<bool TRANS>
__device__ void 
${funcn}_shared(int j, int i1, int offset, 
                const ${dtype}* const __restrict__ b,
                const ${dtype}* const __restrict__ jac,
                ${dtype}* const __restrict__ a
               )
{

    if (TRANS)
    {
        for(int i=0; i<${p*p}; i+=${p})
        {
        //#% for i in range(p):
        % for v in range(nvars):
            a[${v}] += d[i + j]*${flux.src(func=flux_n, name='b', jac=['jac[0]','jac[1]','jac[2]'],
                                           ndims=ndims, sep='+',
                                           idx=[f'{v}', f'{nvars}*(i1 + i)', 'offset'])};
        % endfor
        //#% endfor
        }
    }
    else
    {
        for(int i=0; i<${p}; ++i)
        {
        //#% for i in range(p):
        % for v in range(nvars):
            a[${v}] += d[j*${p}+i]*${flux.src(func=flux_n, name='b', jac=['jac[0]','jac[1]','jac[2]'],
                                              ndims=ndims, sep='+',
                                              idx=[f'{v}', f'{nvars}*(i + i1*{p})', 'offset'])};
        % endfor
        //#% endfor
        }
    }
    return;
}

template<int DIM>
__device__ void
${funcn}_global(int eg, int j, int i1, int i2,
                const ${dtype}* const __restrict__ b,
                int ldb,
                const ${dtype}* const __restrict__ jac,
                ${dtype}* const __restrict__ a
               )
{
    static_assert(DIM < 3, "Invalid global dimension");

    if (DIM == 0)
    {
        for(int i=0; i<${p}; ++i)
        {
        //#% for i in range(p):
        % for v in range(nvars):
            a[${v}] += d[i*${p} + j]*${flux.src(func=flux_n, name='b', jac=['jac[0]','jac[1]','jac[2]'],
                                                ndims=ndims, macro='GLOBAL_IDX',
                                                idx=[f'{v}', 'eg', f'(i + i1*{p} + i2*{p*p})', 'ldb'])};
        % endfor
        //#% endfor
        }
    }
    else if (DIM == 1)
    {
        //for(int i=0; i<${p*p}; i+=${p})
        //{
        % for i in range(p):
        % for v in range(nvars):
            a[${v}] += d[${i*p} + j]*${flux.src(func=flux_n, name='b', jac=['jac[0]','jac[1]','jac[2]'],
                                                ndims=ndims, macro='GLOBAL_IDX',
                                                idx=[f'{v}', 'eg', f'(i1*{p} + {i*p} + i2*{p*p})', 'ldb'])};
        % endfor
        % endfor
        //}
    }
    else if (DIM == 2)
    {
        //for(int i=0; i<${p}; i++)
        //{
        % for i in range(p):
        % for v in range(nvars):
            a[${v}] += d[${i*p} + j]*${flux.src(func=flux_n, name='b', jac=['jac[0]','jac[1]','jac[2]'],
                                                ndims=ndims, macro='GLOBAL_IDX',
                                                idx=[f'{v}', 'eg', f'(i1 + i2*{p} + {i*p*p})', 'ldb'])};
        % endfor
        % endfor
        //}
    }


    return;
}

__device__ void
${funcn}_store(int eg, int j, int i1, int i2, 
               float* const __restrict__ c,
               int ldc,
               const float* const __restrict__ a
              )
{
    for(int v=0; v<${nvars}; ++v)
    {
        c[SOA_IDX(eg, v) + (j + i1*${p} + i2*${p*p})*ldc] = a[v];
    }
    return;
}

__global__ void
__launch_bounds__ (${bcfg.blk_dim})
${funcn}(int n,
         const ${dtype}* const __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{

    // Element No. in global frame
    int eg = ((threadIdx.x % ${bcfg.warp_size}) < ATHRD) ? ABS_E(threadIdx.x, blockIdx.x, blockDim.x) : n;
 
    // Shared Memory Offset
    int el_offset = ${bcfg.shr_size_elem}*LCL_E(threadIdx.x) +
    ((${bcfg.warp_size} - ((${bcfg.shr_offset_rem}*WRP_E(threadIdx.x))%${bcfg.warp_size})) % ${bcfg.warp_size} + WRP_E(threadIdx.x)*${p}) % ${bcfg.warp_size};
 
    extern __shared__ ${dtype} bs[]; // shared array size [elements] = ${bcfg.shr_var_size}

    // z-plane evaluated by thread 
    int thrd_k = ELEM_K(threadIdx.x);  

    ${dtype} acl[${nvars}];

    ${dtype} jacx[] = {1., 0., 0.};
    ${dtype} jacy[] = {0., 1., 0.};
    ${dtype} jacz[] = {0., 0., 1.};

    //unsigned mask = W_MASK(threadIdx.x);

    // n is now number of elements
    if (eg < n)
    {

    % for i_s in range(p): # z-y plane loop

        // Read y-line into register and use that to build z-y plane @ x=${i_s} to shared
    % for j in range(p):
    % for v in range(nvars): # var loop 1
        bs[${v} + TSOA2_IDX(thrd_k, ${j*p}, el_offset)] = b[SOA_IDX(eg, ${v}) + (${i_s} + ${j*p} + ${p*p}*thrd_k)*ldb];
    % endfor # end var loop 1
    % endfor

    % for j in range(p): # y-line loop
    % for v in range(nvars):
        acl[${v}] = 0.;
    % endfor
        // x-line contribution
        ${funcn}_global<0>(eg, ${i_s}, ${j}, thrd_k, b, ldb, jacx, acl);

        // y-line contribution
        ${funcn}_shared<SHR_T>(${j}, thrd_k, el_offset, bs, jacy, acl);
    % if j == 0:
        __syncthreads();
    %endif

        // z-line contribution
        ${funcn}_shared<SHR_N>(thrd_k, ${j}, el_offset, bs, jacz, acl);

        //${funcn}_store(eg, ${i_s}, ${j}, thrd_k, c, ldc, acl);

    % for v in range(nvars):
        c[SOA_IDX(eg, ${v}) + (${i_s + j*p} + thrd_k*${p*p})*ldc] = acl[${v}];
    % endfor

    % endfor # end y-line loop

    % endfor # end z-y plane loop

    }
}

#undef SOA_IDX
#undef GLOBAL_IDX
#undef TSOA2_IDX
#undef ELEM_K
#undef WARP
#undef ATHRD
#undef ABS_E
#undef LCL_E
#undef WRP_E
#undef W_MASK