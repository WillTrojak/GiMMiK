
#define ORDER ${p-1}

#define SHR_SIZE 0
#define WTYPE ${dtype}
#define N_VAR ${nvars}
#define SOA_SZ ${soasz}
#define SOA_IDX(a, v) ((((i) / ${soasz})*${nvars} + (v))*${soasz} + (i) % ${soasz})

__device__ void
${funcn}_tflux(int i, int y0, int ny,
	           const ${dtype}* __restrict__ u, int ldu,
	           ${dtype}* __restrict__ f, int ldf
	           )
{
  
  constexpr int ns = ${p**3};
  
  for (int y = y0; y < y0+ny; y++){
    
    float ftemp[3][13];
    {
      
      float v[] = { u[ldu*y + SOA_IDX(i, 1)], u[ldu*y + SOA_IDX(i, 2)], u[ldu*y + SOA_IDX(i, 3)] };
      float p = u[ldu*y + SOA_IDX(i, 0)];
      
      ftemp[0][0] = 2.5f*v[0];      
      ftemp[0][1] = -0.000625f*u[ldu*y + SOA_IDX(i, 4)] + v[0]*v[0]  + p;
      ftemp[0][2] = -0.000625f*u[ldu*y + SOA_IDX(i, 7)] + v[0]*v[1] ;
      ftemp[0][3] = -0.000625f*u[ldu*y + SOA_IDX(i, 10)] + v[0]*v[2] ;
      ftemp[0][4] = -1000.0f*v[0];
      ftemp[0][7] = -1000.0f*v[1];
      ftemp[0][10] = -1000.0f*v[2];
      
      ftemp[1][0] = 2.5f*v[1];
      ftemp[1][1] = -0.000625f*u[ldu*y + SOA_IDX(i, 5)] + v[1]*v[0] ;
      ftemp[1][2] = -0.000625f*u[ldu*y + SOA_IDX(i, 8)] + v[1]*v[1]  + p;
      ftemp[1][3] = -0.000625f*u[ldu*y + SOA_IDX(i, 11)] + v[1]*v[2] ;
      ftemp[1][5] = -1000.0f*v[0];
      ftemp[1][8] = -1000.0f*v[1];
      ftemp[1][11] = -1000.0f*v[2];
      
      ftemp[2][0] = 2.5f*v[2];
      ftemp[2][1] = -0.000625f*u[ldu*y + SOA_IDX(i, 6)] + v[2]*v[0] ;
      ftemp[2][2] = -0.000625f*u[ldu*y + SOA_IDX(i, 9)] + v[2]*v[1] ;
      ftemp[2][3] = -0.000625f*u[ldu*y + SOA_IDX(i, 12)] + v[2]*v[2]  + p;
      ftemp[2][6] = -1000.0f*v[0];
      ftemp[2][9] = -1000.0f*v[1];
      ftemp[2][12] = -1000.0f*v[2];
      
    };
    
    float sm = 0.00963828554f, ze = 0.f;
    
    f[((0)*ns + y)*ldf + SOA_IDX(i, 0)] = sm*ftemp[0][0] + ze*ftemp[1][0] + ze*ftemp[2][0];
    f[((0)*ns + y)*ldf + SOA_IDX(i, 1)] = sm*ftemp[0][1] + ze*ftemp[1][1] + ze*ftemp[2][1];
    f[((0)*ns + y)*ldf + SOA_IDX(i, 2)] = sm*ftemp[0][2] + ze*ftemp[1][2] + ze*ftemp[2][2];
    f[((0)*ns + y)*ldf + SOA_IDX(i, 3)] = sm*ftemp[0][3] + ze*ftemp[1][3] + ze*ftemp[2][3];
    f[((0)*ns + y)*ldf + SOA_IDX(i, 4)] = sm*ftemp[0][4];
    f[((0)*ns + y)*ldf + SOA_IDX(i, 5)] = ze*ftemp[1][5];
    f[((0)*ns + y)*ldf + SOA_IDX(i, 6)] = ze*ftemp[2][6];
    f[((0)*ns + y)*ldf + SOA_IDX(i, 7)] = sm*ftemp[0][7];
    f[((0)*ns + y)*ldf + SOA_IDX(i, 8)] = ze*ftemp[1][8];
    f[((0)*ns + y)*ldf + SOA_IDX(i, 9)] = ze*ftemp[2][9];
    f[((0)*ns + y)*ldf + SOA_IDX(i, 10)] = sm*ftemp[0][10];
    f[((0)*ns + y)*ldf + SOA_IDX(i, 11)] = ze*ftemp[1][11];
    f[((0)*ns + y)*ldf + SOA_IDX(i, 12)] = ze*ftemp[2][12];
    f[((1)*ns + y)*ldf + SOA_IDX(i, 0)] = ze*ftemp[0][0] + sm*ftemp[1][0] + ze*ftemp[2][0];
    f[((1)*ns + y)*ldf + SOA_IDX(i, 1)] = ze*ftemp[0][1] + sm*ftemp[1][1] + ze*ftemp[2][1];
    f[((1)*ns + y)*ldf + SOA_IDX(i, 2)] = ze*ftemp[0][2] + sm*ftemp[1][2] + ze*ftemp[2][2];
    f[((1)*ns + y)*ldf + SOA_IDX(i, 3)] = ze*ftemp[0][3] + sm*ftemp[1][3] + ze*ftemp[2][3];
    f[((1)*ns + y)*ldf + SOA_IDX(i, 4)] = ze*ftemp[0][4];
    f[((1)*ns + y)*ldf + SOA_IDX(i, 5)] = sm*ftemp[1][5];
    f[((1)*ns + y)*ldf + SOA_IDX(i, 6)] = ze*ftemp[2][6];
    f[((1)*ns + y)*ldf + SOA_IDX(i, 7)] = ze*ftemp[0][7];
    f[((1)*ns + y)*ldf + SOA_IDX(i, 8)] = sm*ftemp[1][8];
    f[((1)*ns + y)*ldf + SOA_IDX(i, 9)] = ze*ftemp[2][9];
    f[((1)*ns + y)*ldf + SOA_IDX(i, 10)] = ze*ftemp[0][10];
    f[((1)*ns + y)*ldf + SOA_IDX(i, 11)] = sm*ftemp[1][11];
    f[((1)*ns + y)*ldf + SOA_IDX(i, 12)] = ze*ftemp[2][12];
    f[((2)*ns + y)*ldf + SOA_IDX(i, 0)] = ze*ftemp[0][0] + ze*ftemp[1][0] + sm*ftemp[2][0];
    f[((2)*ns + y)*ldf + SOA_IDX(i, 1)] = ze*ftemp[0][1] + ze*ftemp[1][1] + sm*ftemp[2][1];
    f[((2)*ns + y)*ldf + SOA_IDX(i, 2)] = ze*ftemp[0][2] + ze*ftemp[1][2] + sm*ftemp[2][2];
    f[((2)*ns + y)*ldf + SOA_IDX(i, 3)] = ze*ftemp[0][3] + ze*ftemp[1][3] + sm*ftemp[2][3];
    f[((2)*ns + y)*ldf + SOA_IDX(i, 4)] = ze*ftemp[0][4];
    f[((2)*ns + y)*ldf + SOA_IDX(i, 5)] = ze*ftemp[1][5];
    f[((2)*ns + y)*ldf + SOA_IDX(i, 6)] = sm*ftemp[2][6];
    f[((2)*ns + y)*ldf + SOA_IDX(i, 7)] = ze*ftemp[0][7];
    f[((2)*ns + y)*ldf + SOA_IDX(i, 8)] = ze*ftemp[1][8];
    f[((2)*ns + y)*ldf + SOA_IDX(i, 9)] = sm*ftemp[2][9];
    f[((2)*ns + y)*ldf + SOA_IDX(i, 10)] = ze*ftemp[0][10];
    f[((2)*ns + y)*ldf + SOA_IDX(i, 11)] = ze*ftemp[1][11];
    f[((2)*ns + y)*ldf + SOA_IDX(i, 12)] = sm*ftemp[2][12];

  }
}

__device__ void
giimik_tfmm_mul_full(int i, int j, int eg,
		     float* __restrict__ f, int ldf,
		     float* __restrict__ c, int ldc
		    )
{
% for k in range(p):
% for v in range(nvars):
	dopt = ${' + '.join('dc[{d} + i]*f[SOA_IDX(eg, {v}) + ({x} + j*{p})*ldf]'.format(p=p, v=v, d=(x*p), x=(x + k*p*p)) for x in range(p))} + 
	       ${' + '.join('dc[{d} + j]*f[SOA_IDX(eg, {v}) + ({x} + i)*ldf]'.format(p=p, v=v, d=(x*p), x=(x*p + k*p*p + p*p*p)) for x in range(p))} + 
	       ${' + '.join('dc[{d}]*f[SOA_IDX(eg, {v}) + ({x} + i + j*{p})*ldf]'.format(p=p, v=v, d=(x*p + k), x=(x*p*p + 2*p*p*p)) for x in range(p))};
	c[SOA_IDX(eg, ${v}) + (i + j*${p} + ${k*p*p})*ldc] = dotp;
% endfor
% endfor
}	

