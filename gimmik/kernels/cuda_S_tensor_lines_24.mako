# -*- coding: utf-8 -*-

/* CHANGE LOG
    p*p threads, with each thread doing lines and all varaibles
    each thread participates in the global load 
*/ 

/*
    NDIMS              = ${ndims}
    ORDER (P)          = ${p-1} (${p})
    NVARS              = ${nvars}
    ELEMENTS PER BLOCK = ${block_elem}

    SHARED USED [B]    = ${shr_size}
    DATA TYPE          = ${dtype}
    SOA_SZ             = ${soasz}
    BLOCK CONFIG       = (${block_elem}, ${p}, ${p})
*/

//#include <cooperative_groups.h>
//#include <cuda/barrier>
//namespace cg = cooperative_groups;

#define SOA_SZ ${soasz}
#define SOA_IDX(i, v) ((((i) / ${soasz})*${nvars} + (v))*${soasz} + (i) % ${soasz})
#define EPB ${block_elem}
#define SSOA_SZ ${block_elem}
#define SHR_NV ${nvars-1}
#define ACC_NV ${nvars-1}

#define BLOCK_DIMX ${block_elem*p*p}
#define SHR_SIZE ${shr_size}
#define ORDER ${p-1}
#define WTYPE ${dtype}
#define N_VAR ${nvars}

#define SHR_IDX_I(e, v, i, j, k) ((e)%SSOA_SZ + ((i) + (j)*${p} + (k)*${p*p})*SSOA_SZ + (v)*SSOA_SZ*${p*p*p})
#define ACC_IDX_I(e, v, i, j, k) ((e)%SSOA_SZ + ((i) + (j)*${p} + (k)*${p*p})*SSOA_SZ + (v)*SSOA_SZ*${p*p*p})


__constant__ ${dtype} dc[${p*p}] = {${','.join(str(D[i,j]) for j in range(p) for i in range(p))} };

__device__ void
${funcn}_import_plane(int i, int j, int k,
                      int eg, ${dtype}* __restrict__ g, int ldg,
                      int el, ${dtype}* __restrict__ s,
                      ${dtype}* __restrict__ a
                     )
{
    ${dtype} r[${nvars}];
    for (int v=0; v<${nvars}; v++)
        r[v] = __ldg(g + SOA_IDX(eg, v) + (i + j*${p} + k*${p*p})*ldg);

    a[ACC_IDX_I(el, 0, i, j, k)] = 0.;
    a[ACC_IDX_I(el, 1, i, j, k)] = 0.;
    a[ACC_IDX_I(el, 2, i, j, k)] = 0.;
    a[ACC_IDX_I(el, 3, i, j, k)] = 0.;
    a[ACC_IDX_I(el, 4, i, j, k)] = r[ 5];
    a[ACC_IDX_I(el, 5, i, j, k)] = r[ 6];
    a[ACC_IDX_I(el, 6, i, j, k)] = r[ 7];
    a[ACC_IDX_I(el, 7, i, j, k)] = 0.;
    a[ACC_IDX_I(el, 8, i, j, k)] = r[ 9];
    a[ACC_IDX_I(el, 9, i, j, k)] = r[10];
    a[ACC_IDX_I(el,10, i, j, k)] = r[11];
    a[ACC_IDX_I(el,11, i, j, k)] = 0.;

    r[ 4] = ${-c['nu']}*r[ 4] + r[0];
    r[ 5] = ${-c['nu']}*r[ 5];
    r[ 6] = ${-c['nu']}*r[ 6];
    r[ 7] = ${-c['nu']}*r[ 7];
    r[ 8] = ${-c['nu']}*r[ 8] + r[0];
    r[ 9] = ${-c['nu']}*r[ 9];
    r[10] = ${-c['nu']}*r[10];
    r[11] = ${-c['nu']}*r[11];
    r[12] = ${-c['nu']}*r[12] + r[0];
                     
    for (int v=0; v<${nvars-1}; v++)
        s[SHR_IDX_I(el, v, i, j, k)] = r[v+1];
      
    return;
}

__device__ void
${funcn}_x(int i, int j, int k, int el,
           ${dtype}* __restrict__ s,
           ${dtype}* __restrict__ a
         )
{
    ${dtype} dotp[] = {${','.join(str(0.) for x in range(nvars-1))}};
    ${dtype} jac[] = {1.,0.,0.};

    for (int x=0; x<${p}; x++)
    {
        ${dtype}4 s0, s1, s2;
        s0.x = s[SHR_IDX_I(el, 0, x, j, k)];
        s0.y = s[SHR_IDX_I(el, 1, x, j, k)];
        s0.z = s[SHR_IDX_I(el, 2, x, j, k)];
        s0.w = s[SHR_IDX_I(el, 3, x, j, k)];
        s1.x = s[SHR_IDX_I(el, 4, x, j, k)];
        s1.y = s[SHR_IDX_I(el, 5, x, j, k)];
        s1.z = s[SHR_IDX_I(el, 6, x, j, k)];
        s1.w = s[SHR_IDX_I(el, 7, x, j, k)];
        s2.x = s[SHR_IDX_I(el, 8, x, j, k)];
        s2.y = s[SHR_IDX_I(el, 9, x, j, k)];
        s2.z = s[SHR_IDX_I(el,10, x, j, k)];
        s2.w = s[SHR_IDX_I(el,11, x, j, k)];
      
        ${dtype} d = dc[x*${p} + i];
        dotp[0] += d*(jac[0]*(s0.x*s0.x + s0.w) +
                      jac[1]*(s0.x*s0.y + s1.x) +
                      jac[2]*(s0.x*s0.z + s1.y));
        dotp[1] += d*(jac[0]*(s0.y*s0.x + s1.z) +
                      jac[1]*(s0.y*s0.y + s1.w) +
                      jac[2]*(s0.y*s0.z + s2.x));
        dotp[2] += d*(jac[0]*(s0.z*s0.x + s2.y) +
                      jac[1]*(s0.z*s0.y + s2.z) +
                      jac[2]*(s0.z*s0.z + s2.w));
        dotp[ 3] += d*jac[0]*s0.x;
        dotp[ 4] += d*jac[1]*s0.x;
        dotp[ 5] += d*jac[2]*s0.x;
        dotp[ 6] += d*jac[0]*s0.y;
        dotp[ 7] += d*jac[1]*s0.y;
        dotp[ 8] += d*jac[2]*s0.y;
        dotp[ 9] += d*jac[0]*s0.z;
        dotp[10] += d*jac[1]*s0.z;
        dotp[11] += d*jac[2]*s0.z;
    }

    for (int v=0; v<${nvars-1}; v++)
        a[ACC_IDX_I(el, v, i, j, k)] += dotp[v];
    
  return;
}

__device__ void
${funcn}_y(int i, int j, int k, int el,
           ${dtype}* __restrict__ s,
           ${dtype}* __restrict__ a
         )
{
    ${dtype} dotp[] = {${','.join(str(0.) for x in range(nvars-1))}};
    ${dtype} jac[] = {0.,1.,0.};

    for (int x=0; x<${p}; x++)
    {
        ${dtype}4 s0, s1, s2;
        s0.x = s[SHR_IDX_I(el, 0, i, x, k)];
        s0.y = s[SHR_IDX_I(el, 1, i, x, k)];
        s0.z = s[SHR_IDX_I(el, 2, i, x, k)];
        s0.w = s[SHR_IDX_I(el, 3, i, x, k)];
        s1.x = s[SHR_IDX_I(el, 4, i, x, k)];
        s1.y = s[SHR_IDX_I(el, 5, i, x, k)];
        s1.z = s[SHR_IDX_I(el, 6, i, x, k)];
        s1.w = s[SHR_IDX_I(el, 7, i, x, k)];
        s2.x = s[SHR_IDX_I(el, 8, i, x, k)];
        s2.y = s[SHR_IDX_I(el, 9, i, x, k)];
        s2.z = s[SHR_IDX_I(el,10, i, x, k)];
        s2.w = s[SHR_IDX_I(el,11, i, x, k)];
        
        ${dtype} d = dc[x*${p} + j];
        dotp[0] += d*(jac[0]*(s0.x*s0.x + s0.w) +
                      jac[1]*(s0.x*s0.y + s1.x) +
                      jac[2]*(s0.x*s0.z + s1.y));
        dotp[1] += d*(jac[0]*(s0.y*s0.x + s1.z) +
                      jac[1]*(s0.y*s0.y + s1.w) +
                      jac[2]*(s0.y*s0.z + s2.x));
        dotp[2] += d*(jac[0]*(s0.z*s0.x + s2.y) +
                      jac[1]*(s0.z*s0.y + s2.z) +
                      jac[2]*(s0.z*s0.z + s2.w));
        dotp[ 3] += d*jac[0]*s0.x;
        dotp[ 4] += d*jac[1]*s0.x;
        dotp[ 5] += d*jac[2]*s0.x;
        dotp[ 6] += d*jac[0]*s0.y;
        dotp[ 7] += d*jac[1]*s0.y;
        dotp[ 8] += d*jac[2]*s0.y;
        dotp[ 9] += d*jac[0]*s0.z;
        dotp[10] += d*jac[1]*s0.z;
        dotp[11] += d*jac[2]*s0.z;
    }

    for (int v=0; v<${nvars-1}; v++)
        a[ACC_IDX_I(el, v, i, j, k)] += dotp[v];
  
  return;
}

__device__ void
${funcn}_z(int i, int j, int k, int el,
          ${dtype}* __restrict__ s,
          ${dtype}* __restrict__ a,
          int eg, 
          ${dtype}* __restrict__ u, int ldu,
          ${dtype}* __restrict__ g, int ldg
         )
{
    ${dtype} dotp[] = {${','.join(str(0.) for x in range(nvars-1))}};
    ${dtype} jac[] = {0.,0.,1.};

    ${dtype} s0 = __ldcg(u + SOA_IDX(eg, 4) + (i + j*${p} + k*${p*p})*ldu);
    ${dtype} s4 = __ldcg(u + SOA_IDX(eg, 8) + (i + j*${p} + k*${p*p})*ldu);
    ${dtype} s8 = __ldcg(u + SOA_IDX(eg,12) + (i + j*${p} + k*${p*p})*ldu);

    for (int x=0; x<${p}; x++)
    {
        ${dtype}4 s0, s1, s2;
        s0.x = s[SHR_IDX_I(el, 0, i, j, x)]; // u
        s0.y = s[SHR_IDX_I(el, 1, i, j, x)]; // v
        s0.z = s[SHR_IDX_I(el, 2, i, j, x)]; // w
        s0.w = s[SHR_IDX_I(el, 3, i, j, x)]; // qx
        s1.x = s[SHR_IDX_I(el, 4, i, j, x)]; // qy 
        s1.y = s[SHR_IDX_I(el, 5, i, j, x)]; // qz
        s1.z = s[SHR_IDX_I(el, 6, i, j, x)]; // rx
        s1.w = s[SHR_IDX_I(el, 7, i, j, x)]; // ry
        s2.x = s[SHR_IDX_I(el, 8, i, j, x)]; // rz
        s2.y = s[SHR_IDX_I(el, 9, i, j, x)]; // sx
        s2.z = s[SHR_IDX_I(el,10, i, j, x)]; // sy
        s2.w = s[SHR_IDX_I(el,11, i, j, x)]; // sz
        
        ${dtype} d = dc[x*${p} + k];
        
        dotp[0] += d*(jac[0]*(s0.x*s0.x + s0.w) +
                      jac[1]*(s0.x*s0.y + s1.x) +
                      jac[2]*(s0.x*s0.z + s1.y));
        dotp[1] += d*(jac[0]*(s0.y*s0.x + s1.z) +
                      jac[1]*(s0.y*s0.y + s1.w) +
                      jac[2]*(s0.y*s0.z + s2.x));
        dotp[2] += d*(jac[0]*(s0.z*s0.x + s2.y) +
                      jac[1]*(s0.z*s0.y + s2.z) +
                      jac[2]*(s0.z*s0.z + s2.w));
        dotp[ 3] += d*jac[0]*s0.x;
        dotp[ 4] += d*jac[1]*s0.x;
        dotp[ 5] += d*jac[2]*s0.x;
        dotp[ 6] += d*jac[0]*s0.y;
        dotp[ 7] += d*jac[1]*s0.y;
        dotp[ 8] += d*jac[2]*s0.y;
        dotp[ 9] += d*jac[0]*s0.z;
        dotp[10] += d*jac[1]*s0.z;
        dotp[11] += d*jac[2]*s0.z;
    }

    g[SOA_IDX(eg, 0) + (i + j*${p} + k*${p*p})*ldg] = ${c['ac-zeta']}*(dotp[3] + dotp[7] + dotp[11] + a[ACC_IDX_I(el, 3, i, j, k)] + a[ACC_IDX_I(el, 7, i, j, k)] + a[ACC_IDX_I(el, 11, i, j, k)]);
% for v in range(ndims):
    g[SOA_IDX(eg, ${v+1}) + (i + j*${p} + k*${p*p})*ldg] = dotp[${v}] + a[ACC_IDX_I(el, ${v}, i, j, k)];
% endfor
% for v in range(ndims*ndims):
% if v == 0 or v == 4 or v == 8:
    g[SOA_IDX(eg, ${v+4}) + (i + j*${p} + k*${p*p})*ldg] = ${-1/c['tr']}*(dotp[${v+3}] + a[ACC_IDX_I(el, ${v+3}, i, j, k)] + s${v});
% else:
    g[SOA_IDX(eg, ${v+4}) + (i + j*${p} + k*${p*p})*ldg] = ${-1/c['tr']}*(dotp[${v+3}] + a[ACC_IDX_I(el, ${v+3}, i, j, k)]);
% endif
% endfor
}

__global__ void
${funcn}(int n,
        ${dtype}* __restrict__ b, int ldb,
        ${dtype}* __restrict__ c, int ldc
       )
{
    //using barrier = cuda::barrier<cuda::thread_scope_block>;
    //__shared__  barrier bar;
    //auto block = cg::this_thread_block();

    extern __shared__ ${dtype} s[];

    int ti = (threadIdx.x/${block_elem}) % ${p};
    int tj = (threadIdx.x/${block_elem*p}) % ${p};
    int tk = threadIdx.x % ${block_elem};
  
    int oss = 0;
    int osa = ${block_elem*(nvars-1)*p*p*p};
    int el = tk;
    int eg = blockIdx.x*${block_elem} + el;


    //if (block.thread_rank() == 0)
    //    init(&bar, (min((blockIdx.x+1)*${block_elem}, n) - blockIdx.x*${block_elem})*${p*p});
    //block.sync();

    if(eg < n)
    {
        for(int k=0; k<${p}; k++)
        {
            ${funcn}_import_plane(ti, tj, k, eg, b, ldb, el, s + oss, s + osa);
            //bar.arrive_and_wait();
            __syncthreads();
            ${funcn}_x(ti, tj, k, el, s + oss, s + osa);
            ${funcn}_y(ti, tj, k, el, s + oss, s + osa);
        }
        //bar.arrive_and_wait();
        __syncthreads();

        for(int k=0; k<${p}; k++)
            ${funcn}_z(ti, tj, k, el, s + oss, s + osa, eg, b, ldb, c, ldc);
    }

  return;
}

#undef SHR_IDX_I
#undef ACC_IDX_I
#undef ACC_NV
#undef SHR_NV
#undef SSOA_SZ