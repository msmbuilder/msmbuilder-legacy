#ifndef _DEVICE_RMSD_H_
#define _DEVICE_RMSD_H_

#ifndef numDims
#define numDims 3
#endif

/** \brief Rotate the entire set of conformations using corresponding set of torsion angles
  * \param[in] numConfs Number of conformers
  * \param[in] numAtoms Number of Atoms
  * \param[in,out] g_X the conformers that will be rotated
  * \param[in,out] g_R the conformers that will be rotated
  */
__global__ void k_rotate_all( uint numConfs, uint numAtoms, float *g_X, float *g_R ) {

    uint const t = blockDim.x*blockIdx.x + threadIdx.x;

    if( t < numConfs ) {

        for(int i=0;i<numAtoms;i++) {

            float i_x = g_X[i*numConfs*numDims+0*numConfs+t];
            float i_y = g_X[i*numConfs*numDims+1*numConfs+t];
            float i_z = g_X[i*numConfs*numDims+2*numConfs+t];
        
            float new_x = g_R[0*numConfs + t]*i_x + g_R[1*numConfs + t]*i_y + g_R[2*numConfs + t]*i_z;
            float new_y = g_R[3*numConfs + t]*i_x + g_R[4*numConfs + t]*i_y + g_R[5*numConfs + t]*i_z;
            float new_z = g_R[6*numConfs + t]*i_x + g_R[7*numConfs + t]*i_y + g_R[8*numConfs + t]*i_z;

            g_X[i*numConfs*numDims+0*numConfs+t] = new_x;
            g_X[i*numConfs*numDims+1*numConfs+t] = new_y;
            g_X[i*numConfs*numDims+2*numConfs+t] = new_z;
        
        }

    }

}

/** \brief Center all the conformers so that each conformer has their center of mass set
  * to the origin.
  * \param[in] numConfs Number of conformers
  * \param[in] numAtoms Number of Atoms
  * \param[in,out] g_X the conformers that will be centered
  * \param[in] g_subset_flags an optional boolean flag that determines whether or not to keep an atom
  */
__global__ void k_center_conformers( uint numConfs, uint numAtoms, float *g_X, int *g_subset_flags = NULL) {

    uint const t = blockDim.x*blockIdx.x + threadIdx.x;

    if( t < numConfs ) {

        float center_offset_x = 0;
        float center_offset_y = 0;
        float center_offset_z = 0;

        for(int i=0; i<numAtoms; i++) {
            if( g_subset_flags != NULL ) {
                if(g_subset_flags[i] == true) {
                    center_offset_x += g_X[i*numConfs*numDims+0*numConfs+t];
                    center_offset_y += g_X[i*numConfs*numDims+1*numConfs+t];
                    center_offset_z += g_X[i*numConfs*numDims+2*numConfs+t];
                }
            }
        }
        center_offset_x /= (float) numAtoms;
        center_offset_y /= (float) numAtoms;
        center_offset_z /= (float) numAtoms;
        for(int i=0; i<numAtoms; i++) {
            g_X[i*numConfs*numDims+0*numConfs+t] -= center_offset_x;
            g_X[i*numConfs*numDims+1*numConfs+t] -= center_offset_y;
            g_X[i*numConfs*numDims+2*numConfs+t] -= center_offset_z;
        }

    }
}

/** \brief Pre-compute the inner product, as it is re-used many times
  * \param[in] numConfs Number of conformers
  * \param[in] numAtoms Number of Atoms
  * \param[in,out] g_X the conformers that will be centered
  * \param[in] g_subset_flags an optional boolean flag that determines whether or not to keep an atom
  */
__global__ void k_precompute_G( uint numConfs, uint numAtoms, float *g_confs, float *d_G_, int *g_subset_flags = NULL ) {

    uint const t = blockDim.x*blockIdx.x + threadIdx.x;
    double g_ = 0;

    if( t < numConfs) {
        for(uint i=0;i<numAtoms;i++) {
            if( g_subset_flags != NULL ) {
                if(g_subset_flags[i] == true) {
                    double r_x = g_confs[i*numConfs*numDims+0*numConfs+t];
                    double r_y = g_confs[i*numConfs*numDims+1*numConfs+t];
                    double r_z = g_confs[i*numConfs*numDims+2*numConfs+t];
           
                    g_ += r_x * r_x 
                        + r_y * r_y
                        + r_z * r_z;
                }
            }
        }
        d_G_[t] = (float) g_;
    }

}

// taken from Kyle Beauchamp's code
__inline__ __device__
int solve_cubic_equation(double  c3, double  c2,  double c1, double c0,
                         double *x1, double *x2, double *x3)
{
  double a2 = c2/c3;
  double a1 = c1/c3;
  double a0 = c0/c3;

  double q = a1/3.0 - a2*a2/9.0;
  double r = (a1*a2 - 3.0*a0)/6.0 - a2*a2*a2 / 27.0;
  double delta = q*q*q + r*r;

  if (delta>0.0)
    {
      double s1 = r + sqrt(delta);
      s1 = (s1>=0.0) ? pow(s1,1./3.) : -pow(-s1,1./3.);

      double s2 = r - sqrt(delta);
      s2 = (s2>=0.0) ? pow(s2,1./3.) : -pow(-s2,1./3.);

      *x1 = (s1+s2) - a2/3.0;
      *x2 = *x3 = -0.5 * (s1+s2) - a2/3.0;

      return 1;
    }
  else if (delta < 0.0)
    {
      double theta = acos(r/sqrt(-q*q*q)) / 3.0;
      double costh = cos(theta);
      double sinth = sin(theta);
      double sq = sqrt(-q);

      *x1 = 2.0*sq*costh - a2/3.0;
      *x2 = -sq*costh - a2/3.0 - sqrt(3.) * sq * sinth;
      *x3 = -sq*costh - a2/3.0 + sqrt(3.) * sq * sinth;

      return 3;
    }
  else
    {
      double s = (r>=0.0) ? pow(r,1./3.) : -pow(-r,1./3.);
      *x1 = 2.0*s - a2/3.0;
      *x2 = *x3 = -s - a2/3.0;

      return 3;
    }
}
__inline__ __device__
int quartic_equation_solve_exact(double *r1, double *r2, double *r3, double *r4,
				 int *nr12, int *nr34,double d0,double d1,double d2, double d3, double d4)
{
  double a3 = d3/d4;
  double a2 = d2/d4;
  double a1 = d1/d4;
  double a0 = d0/d4;

  double au2 = -a2;
  double au1 = (a1*a3 - 4.0*a0) ;
  double au0 = 4.0*a0*a2 - a1*a1 - a0*a3*a3;

  double x1, x2, x3;
  int nr = solve_cubic_equation(1.0, au2, au1, au0, &x1, &x2, &x3);

  double u1;
  if (nr==1) u1 = x1;
  else u1 = (x1>x3) ? x1 : x3;

  double R2 = 0.25*a3*a3 + u1 - a2;
  double R = (R2>0.0) ? sqrt(R2) : 0.0;

  double D2, E2;
  if (R != 0.0)
    {
      double foo1 = 0.75*a3*a3 - R2 - 2.0*a2;
      double foo2 = 0.25*(4.0*a3*a2 - 8.0*a1 - a3*a3*a3) / R;
      D2 = foo1 + foo2;
      E2 = foo1 - foo2;
    }
  else
    {
      double foo1 = 0.75*a3*a3 - 2.0*a2;
      double foo2 = 2.0 * sqrt(u1*u1 - 4.0*a0);
      D2 = foo1 + foo2;
      E2 = foo1 - foo2;
    }

  if (D2 >= 0.0)
    {
      double D = sqrt(D2);
      *r1 = -0.25*a3 + 0.5*R - 0.5*D;
      *r2 = -0.25*a3 + 0.5*R + 0.5*D;
      *nr12 = 2;
    }
  else
    {
      *r1 = *r2 = -0.25*a3 + 0.5*R;
      *nr12 = 0;
    }

  if (E2 >= 0.0)
    {
      double E = sqrt(E2);
      *r3 = -0.25*a3 - 0.5*R - 0.5*E;
      *r4 = -0.25*a3 - 0.5*R + 0.5*E;
      *nr34 = 2;
    }
  else
    {
      *r3 = *r4 = -0.25*a3 - 0.5*R;
      *nr34 = 0;
    }
  return *nr12 + *nr34;
}




__inline__ __device__ float DirectSolve(float lambda, float C_0, float C_1, float C_2)
{
  double result;
  double r1,r2,r3,r4;
  int nr1,nr2;
  quartic_equation_solve_exact(&r1,&r2,&r3,&r4,&nr1,&nr2,(double )C_0,(double)C_1,(double)C_2,0.0,1.0);
  result=max(r1,r2);
  result=max(result,r3);
  result=max(result,r4);
  
  return(result);
}

// be careful to also apply the g_subset_flags to the G_ matrix
// try restricting later one
__device__ float calc_rmsd2_goodenough( uint const numAtoms, uint numConfs,
                             float const *d_X_, uint ctr_ID, uint pointID, 
                             float const *G_, float *g_rot_mat = NULL ) {

    double m00,m01,m02,
           m10,m11,m12,
           m20,m21,m22;
          
           m00=m01=m02=m10=m11=m12=m20=m21=m22=0;

    double G_A = G_[ctr_ID]; 
    double G_B = G_[pointID];
    
    for(uint i=0;i<numAtoms;i++) {
        float r_temp_Bx = d_X_[i*numConfs*numDims+0*numConfs+pointID];
        float r_temp_By = d_X_[i*numConfs*numDims+1*numConfs+pointID];
        float r_temp_Bz = d_X_[i*numConfs*numDims+2*numConfs+pointID];
   
        float r_temp_Ax = d_X_[i*numConfs*numDims+0*numConfs+ctr_ID];
        float r_temp_Ay = d_X_[i*numConfs*numDims+1*numConfs+ctr_ID];
        float r_temp_Az = d_X_[i*numConfs*numDims+2*numConfs+ctr_ID];

        m00 += r_temp_Bx * r_temp_Ax;
        m01 += r_temp_Bx * r_temp_Ay;
        m02 += r_temp_Bx * r_temp_Az;

        m10 += r_temp_By * r_temp_Ax; 
        m11 += r_temp_By * r_temp_Ay; 
        m12 += r_temp_By * r_temp_Az; 

        m20 += r_temp_Bz * r_temp_Ax; 
        m21 += r_temp_Bz * r_temp_Ay; 
        m22 += r_temp_Bz * r_temp_Az; 
    }
   
    double C_2 = m00*m00 + m01*m01 + m02*m02
              + m10*m10 + m11*m11 + m12*m12
              + m20*m20 + m21*m21 + m22*m22;
          C_2 *= -2.0; 
    
    float k00 = m00 + m11 + m22;
    float k01 = m12 - m21;
    float k02 = m20 - m02;
    float k03 = m01 - m10;

    //float k10 = k01;
    float k11 = m00 - m11 - m22;
    float k12 = m01 + m10;
    float k13 = m20 + m02;

    //float k20 = k02;
    //float k21 = k12;
    float k22 =-m00 + m11 - m22;
    float k23 = m12 + m21;

    //float k30 = k03;
    //float k31 = k13;
    //float k32 = k23;
    float k33 =-m00 - m11 + m22;

    
    // Determinant by Laplacian expansion
    // note that determinant calculations are prone to doubleing point errors
    // we need to find some way to minimize this!
    double detK = k01*k01*k23*k23   - k22*k33*k01*k01   + 2*k33*k01*k02*k12
               - 2*k01*k02*k13*k23 - 2*k01*k03*k12*k23 + 2*k22*k01*k03*k13
               + k02*k02*k13*k13   - k11*k33*k02*k02   - 2*k02*k03*k12*k13
               + 2*k11*k02*k03*k23 + k03*k03*k12*k12   - k11*k22*k03*k03
               - k00*k33*k12*k12   + 2*k00*k12*k13*k23 - k00*k22*k13*k13
               - k00*k11*k23*k23   + k00*k11*k22*k33;

    double C_1 = -8.0 * (m00 * (m11*m22-m12*m21) - m01*(m10*m22-m12*m20) + m02*(m10*m21-m11*m20));

    // Doubles Newton-Raphson
    
    double lambda_old, lambda2, a, b;

    double lambda = (G_A + G_B) / 2.0;
    uint  maxits = 25;
    float tolerance = 1.0e-6f;

    // thread divergence
    // switch to exact solver
    for (int i = 0; i < maxits; i++)
    {
        lambda_old = lambda;
        lambda2 = lambda_old * lambda_old;// lambda^2
        b = (lambda2 + C_2) * lambda_old; // b = lambda_old^3 + C_2*lambda_old
        a = b + C_1;                      // a = lambda_old^3 + C_2*lambda_old + C_1
        lambda = lambda_old - (a * lambda_old + detK) / (2.0 * lambda2 * lambda_old + b + a);
        //     = lambda_old - lambda_old^4 + C_2*lambda_old^2 + C_1*lambda_old + C_0
        //                    ------------------------------------------------------
        //                            4 * lambda^3 + 2 C_2*lambda^2 + C_1
        if (fabs(lambda - lambda_old) < fabs( tolerance * lambda)) break;
    }
    
    /* Quartic Solvers 
    double lambda = (G_A+G_B)/2.0;
   
    lambda = DirectSolve(lambda, detK, C_1, C_2);
    */

    double rmsd2 =  (G_A + G_B - 2.0 * lambda) / numAtoms;

    // check for rotation matrix here!
    // NOTE* this still needs to be implemented (AND NOT COMPLETE)	
    // we MUST find lambda before we can calculate the rotation matrix
    
    // do these registers get allocated even if g_rot_mat is empty?
    
    // uses an extra 5 registers, may need to move to separate function
    /* enable when needed
    if( g_rot_mat != NULL ) {

        float SxzpSzx = m02 + m20;
        float SyzpSzy = m12 + m21;
        float SxypSyx = m01 + m10;
        float SyzmSzy = m12 - m21;
        float SxzmSzx = m02 - m20;
        float SxymSyx = m01 - m10;
        float SxxpSyy = m00 + m11;
        float SxxmSyy = m00 - m11;

        float a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34,a41,a42,a43,a44;
        float q1,q2,q3,q4;
        double a2, x2, y2, z2; 
        double xy, az, zx, ay, yz, ax; 
        double a3344_4334, a3244_4234, a3243_4233, a3143_4133,a3144_4134, a3142_4132; 

        float normq,qsqr;

        a11 = SxxpSyy + m22 - lambda; a12 = SyzmSzy; a13 = - SxzmSzx; a14 = SxymSyx;
        a21 = SyzmSzy; a22 = SxxmSyy - m22-lambda; a23 = SxypSyx; a24= SxzpSzx;
        a31 = a13; a32 = a23; a33 = m11-m00-m22 - lambda; a34 = SyzpSzy;
        a41 = a14; a42 = a24; a43 = a34; a44 = m22 - SxxpSyy - lambda;
        a3344_4334 = a33 * a44 - a43 * a34; a3244_4234 = a32 * a44-a42*a34;
        a3243_4233 = a32 * a43 - a42 * a33; a3143_4133 = a31 * a43-a41*a33;
        a3144_4134 = a31 * a44 - a41 * a34; a3142_4132 = a31 * a42-a41*a32;
        q1 =  a22*a3344_4334-a23*a3244_4234+a24*a3243_4233;
        q2 = -a21*a3344_4334+a23*a3144_4134-a24*a3143_4133;
        q3 =  a21*a3244_4234-a22*a3144_4134+a24*a3142_4132;
        q4 = -a21*a3243_4233+a22*a3143_4133-a23*a3142_4132;

        qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;
     
        normq = sqrt(qsqr);
        q1 /= normq;
        q2 /= normq;
        q3 /= normq;
        q4 /= normq;

        a2 = q1 * q1;
        x2 = q2 * q2;
        y2 = q3 * q3;
        z2 = q4 * q4;

        xy = q2 * q3;
        az = q1 * q4;
        zx = q4 * q2;
        ay = q1 * q3;
        yz = q3 * q4;
        ax = q1 * q2;

        //coalesced
        g_rot_mat[0*numConfs + pointID] = a2 + x2 - y2 - z2;
        g_rot_mat[1*numConfs + pointID] = 2 * (xy + az);
        g_rot_mat[2*numConfs + pointID] = 2 * (zx - ay);
        g_rot_mat[3*numConfs + pointID] = 2 * (xy - az);
        g_rot_mat[4*numConfs + pointID] = a2 - x2 + y2 - z2;
        g_rot_mat[5*numConfs + pointID] = 2 * (yz + ax);
        g_rot_mat[6*numConfs + pointID] = 2 * (zx + ay);
        g_rot_mat[7*numConfs + pointID] = 2 * (yz - ax);
        g_rot_mat[8*numConfs + pointID] = a2 - x2 - y2 + z2;

    }
    */
    // just return rmsd2
    return max( (float) 0, rmsd2);
}


// This kernel does three things:
// 1. It finds the eigenvalue for the rotation quaternion (as above). 
//    However, the user can specify which atoms to take into account.
//    This is done via g_subset_flags
// 2. It then recovers the rotation matrix
// 3. It then applies the rotation matrix onto every Atom
// Named after "Lee-Ping"
// This is the fusion of several kernels mainly to avoid read into global memory
__device__ float calc_fused_lprmsd( uint const numAtoms, uint numConfs,
                             float const *d_X_, uint ctr_ID, uint pointID, 
                             float const *G_, int const *g_subset_flags) {

    // move lambda finding business to an inline function
    double m00,m01,m02,
           m10,m11,m12,
           m20,m21,m22;
          
           m00=m01=m02=m10=m11=m12=m20=m21=m22=0;

    double G_A = G_[ctr_ID]; 
    double G_B = G_[pointID];
    
    for(uint i=0;i<numAtoms;i++) {
            if(g_subset_flags[i] == true) {
                float r_temp_Bx = d_X_[i*numConfs*numDims+0*numConfs+pointID];
                float r_temp_By = d_X_[i*numConfs*numDims+1*numConfs+pointID];
                float r_temp_Bz = d_X_[i*numConfs*numDims+2*numConfs+pointID];
           
                float r_temp_Ax = d_X_[i*numConfs*numDims+0*numConfs+ctr_ID];
                float r_temp_Ay = d_X_[i*numConfs*numDims+1*numConfs+ctr_ID];
                float r_temp_Az = d_X_[i*numConfs*numDims+2*numConfs+ctr_ID];

                m00 += r_temp_Bx * r_temp_Ax;
                m01 += r_temp_Bx * r_temp_Ay;
                m02 += r_temp_Bx * r_temp_Az;

                m10 += r_temp_By * r_temp_Ax; 
                m11 += r_temp_By * r_temp_Ay; 
                m12 += r_temp_By * r_temp_Az; 

                m20 += r_temp_Bz * r_temp_Ax; 
                m21 += r_temp_Bz * r_temp_Ay; 
                m22 += r_temp_Bz * r_temp_Az; 
        }
    }
   
    double C_2 = m00*m00 + m01*m01 + m02*m02
              + m10*m10 + m11*m11 + m12*m12
              + m20*m20 + m21*m21 + m22*m22;
          C_2 *= -2.0; 
    
    float k00 = m00 + m11 + m22;
    float k01 = m12 - m21;
    float k02 = m20 - m02;
    float k03 = m01 - m10;

    //float k10 = k01;
    float k11 = m00 - m11 - m22;
    float k12 = m01 + m10;
    float k13 = m20 + m02;

    //float k20 = k02;
    //float k21 = k12;
    float k22 =-m00 + m11 - m22;
    float k23 = m12 + m21;

    //float k30 = k03;
    //float k31 = k13;
    //float k32 = k23;
    float k33 =-m00 - m11 + m22;

    
    // Determinant by Laplacian expansion
    // note that determinant calculations are prone to doubleing point errors
    // we need to find some way to minimize this!
    double detK = k01*k01*k23*k23   - k22*k33*k01*k01   + 2*k33*k01*k02*k12
               - 2*k01*k02*k13*k23 - 2*k01*k03*k12*k23 + 2*k22*k01*k03*k13
               + k02*k02*k13*k13   - k11*k33*k02*k02   - 2*k02*k03*k12*k13
               + 2*k11*k02*k03*k23 + k03*k03*k12*k12   - k11*k22*k03*k03
               - k00*k33*k12*k12   + 2*k00*k12*k13*k23 - k00*k22*k13*k13
               - k00*k11*k23*k23   + k00*k11*k22*k33;

    double C_1 = -8.0 * (m00 * (m11*m22-m12*m21) - m01*(m10*m22-m12*m20) + m02*(m10*m21-m11*m20));

    // Doubles Newton-Raphson
    
    double lambda_old, lambda2, a, b;

    double lambda = (G_A + G_B) / 2.0;
    uint  maxits = 25;
    float tolerance = 1.0e-6f;

    // thread divergence
    // switch to exact solver
    for (int i = 0; i < maxits; i++)
    {
        lambda_old = lambda;
        lambda2 = lambda_old * lambda_old;// lambda^2
        b = (lambda2 + C_2) * lambda_old; // b = lambda_old^3 + C_2*lambda_old
        a = b + C_1;                      // a = lambda_old^3 + C_2*lambda_old + C_1
        lambda = lambda_old - (a * lambda_old + detK) / (2.0 * lambda2 * lambda_old + b + a);
        //     = lambda_old - lambda_old^4 + C_2*lambda_old^2 + C_1*lambda_old + C_0
        //                    ------------------------------------------------------
        //                            4 * lambda^3 + 2 C_2*lambda^2 + C_1
        if (fabs(lambda - lambda_old) < fabs( tolerance * lambda)) break;
    }
    
    /* Quartic Solvers 
    double lambda = (G_A+G_B)/2.0;
   
    lambda = DirectSolve(lambda, detK, C_1, C_2);
    */

    double rmsd2 =  (G_A + G_B - 2.0 * lambda) / numAtoms;

    // recover rotation matrix
    double g0,g1,g2,g3,g4,g5,g6,g7,g8;


    double SxzpSzx = m02 + m20;
    double SyzpSzy = m12 + m21;
    double SxypSyx = m01 + m10;
    double SyzmSzy = m12 - m21;
    double SxzmSzx = m02 - m20;
    double SxymSyx = m01 - m10;
    double SxxpSyy = m00 + m11;
    double SxxmSyy = m00 - m11;

    double a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34,a41,a42,a43,a44;
    double q1,q2,q3,q4;
    double a2, x2, y2, z2; 
    double xy, az, zx, ay, yz, ax; 
    double a3344_4334, a3244_4234, a3243_4233, a3143_4133,a3144_4134, a3142_4132; 

    double normq,qsqr;

    a11 = SxxpSyy + m22 - lambda; a12 = SyzmSzy; a13 = - SxzmSzx; a14 = SxymSyx;
    a21 = SyzmSzy; a22 = SxxmSyy - m22-lambda; a23 = SxypSyx; a24= SxzpSzx;
    a31 = a13; a32 = a23; a33 = m11-m00-m22 - lambda; a34 = SyzpSzy;
    a41 = a14; a42 = a24; a43 = a34; a44 = m22 - SxxpSyy - lambda;
    a3344_4334 = a33 * a44 - a43 * a34; a3244_4234 = a32 * a44-a42*a34;
    a3243_4233 = a32 * a43 - a42 * a33; a3143_4133 = a31 * a43-a41*a33;
    a3144_4134 = a31 * a44 - a41 * a34; a3142_4132 = a31 * a42-a41*a32;
    q1 =  a22*a3344_4334-a23*a3244_4234+a24*a3243_4233;
    q2 = -a21*a3344_4334+a23*a3144_4134-a24*a3143_4133;
    q3 =  a21*a3244_4234-a22*a3144_4134+a24*a3142_4132;
    q4 = -a21*a3243_4233+a22*a3143_4133-a23*a3142_4132;

    qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;

/*
    double evecprec = 1e-6;

    if (qsqr < evecprec)
    {
        q1 =  a12*a3344_4334 - a13*a3244_4234 + a14*a3243_4233;
        q2 = -a11*a3344_4334 + a13*a3144_4134 - a14*a3143_4133;
        q3 =  a11*a3244_4234 - a12*a3144_4134 + a14*a3142_4132;
        q4 = -a11*a3243_4233 + a12*a3143_4133 - a13*a3142_4132;
        qsqr = q1*q1 + q2 *q2 + q3*q3+q4*q4;

        if (qsqr < evecprec)
        {
            double a1324_1423 = a13 * a24 - a14 * a23, a1224_1422 = a12 * a24 - a14 * a22;
            double a1223_1322 = a12 * a23 - a13 * a22, a1124_1421 = a11 * a24 - a14 * a21;
            double a1123_1321 = a11 * a23 - a13 * a21, a1122_1221 = a11 * a22 - a12 * a21;

            q1 =  a42 * a1324_1423 - a43 * a1224_1422 + a44 * a1223_1322;
            q2 = -a41 * a1324_1423 + a43 * a1124_1421 - a44 * a1123_1321;
            q3 =  a41 * a1224_1422 - a42 * a1124_1421 + a44 * a1122_1221;
            q4 = -a41 * a1223_1322 + a42 * a1123_1321 - a43 * a1122_1221;
            qsqr = q1*q1 + q2 *q2 + q3*q3+q4*q4;

            if (qsqr < evecprec)
            {
                q1 =  a32 * a1324_1423 - a33 * a1224_1422 + a34 * a1223_1322;
                q2 = -a31 * a1324_1423 + a33 * a1124_1421 - a34 * a1123_1321;
                q3 =  a31 * a1224_1422 - a32 * a1124_1421 + a34 * a1122_1221;
                q4 = -a31 * a1223_1322 + a32 * a1123_1321 - a33 * a1122_1221;
                qsqr = q1*q1 + q2 *q2 + q3*q3 + q4*q4;
                
                if (qsqr < evecprec)
                {
                    g0 = g4 = g8 = 1.0;
                    g1 = g2 = g3 = g5 = g6 = g7 = 0.0;

                    return(0);
                }
            }
        }
    }
*/
 
    normq = sqrt(qsqr);
    q1 /= normq;
    q2 /= normq;
    q3 /= normq;
    q4 /= normq;

    a2 = q1 * q1;
    x2 = q2 * q2;
    y2 = q3 * q3;
    z2 = q4 * q4;

    xy = q2 * q3;
    az = q1 * q4;
    zx = q4 * q2;
    ay = q1 * q3;
    yz = q3 * q4;
    ax = q1 * q2;

    // calculate rotation matrix
    g0 = a2 + x2 - y2 - z2;
    g1 = 2 * (xy + az);
    g2 = 2 * (zx - ay);
    g3 = 2 * (xy - az);
    g4 = a2 - x2 + y2 - z2;
    g5 = 2 * (yz + ax);
    g6 = 2 * (zx + ay);
    g7 = 2 * (yz - ax);
    g8 = a2 - x2 - y2 + z2;

    double euc_distance = 0;

    // apply rotation matrix to each atom
    // and calculate new euclidean distance
    // this is fused into a single kernel so we dont have to offload
    // the rot_matrix values or launch a new Euc distance
    // this also has a fairly high ILP of roughly 3.0
    for(int i=0;i<numAtoms;i++) {

        // a bit overlap as we've already loaded current coorindates once
        float i_x = d_X_[i*numConfs*numDims+0*numConfs+ctr_ID];
        float i_y = d_X_[i*numConfs*numDims+1*numConfs+ctr_ID];
        float i_z = d_X_[i*numConfs*numDims+2*numConfs+ctr_ID];
    
        // apply rotation matrix
        float new_x = g0*i_x + g1*i_y + g2*i_z;
        float new_y = g3*i_x + g4*i_y + g5*i_z;
        float new_z = g6*i_x + g7*i_y + g8*i_z;

        /* don't need to apply it directly to gpu memory
        d_X_[i*numConfs*numDims+0*numConfs+pointID] = new_x;
        d_X_[i*numConfs*numDims+1*numConfs+pointID] = new_y;
        d_X_[i*numConfs*numDims+2*numConfs+pointID] = new_z;
        */

        float pointAx = d_X_[i*numConfs*numDims+0*numConfs+pointID];
        float pointAy = d_X_[i*numConfs*numDims+1*numConfs+pointID];
        float pointAz = d_X_[i*numConfs*numDims+2*numConfs+pointID];

        // calculate the new Euclidean Distance
        euc_distance += (new_x - pointAx) * (new_x - pointAx) +
                        (new_y - pointAy) * (new_y - pointAy) +
                        (new_z - pointAz) * (new_z - pointAz);
    }

    euc_distance /= numAtoms; 

    return euc_distance;
}

/** \brief Center all the conformers so that each conformer has their center of mass set
  * to the origin.
  * \param[in] numAtoms number of atoms
  * \param[in] numConfs number of conformers
  * \param[in] ctr_ID which conformations we're comparing against
  * \param[in] d_X storage array of all the conformations stored in
               d_X[A * numConfs * numDims + D * numConfs + P]
               to access the Ath atom of the Dth dimension of the Pth conformations
  * \param[out] rmsds
  * \param[in] pre-computed inner products
  */ 
__global__ static void 
k_all_against_one_rmsd(uint const numAtoms, uint const numConfs, uint const ctr_ID, 
                       float const * d_X, float *d_rmsds, float const *G_) {

    uint tid = threadIdx.x;
    uint t = blockDim.x*blockIdx.x + tid;

    if(t < numConfs) {

        d_rmsds[t] = sqrt(calc_rmsd2_goodenough( numAtoms, numConfs, d_X, ctr_ID, t, G_));

        // Error-testing: compare each conformer to itself.
        // This should return 0.   
        // d_rmsds[t] = calc_rmsd2_goodenough( numAtoms, numConfs, d_X, t, t, G_ );
    }

}

/** Same as above, except we consider only a subset of the atoms when calculating RMSD
    input via d_subset_flag */
__global__ static void 
k_all_against_one_lprmsd(uint const numAtoms, uint const numConfs, uint const ctr_ID, 
                       float const * d_X, float *d_rmsds, float *G_, int const *d_subset_flag_) {

    uint tid = threadIdx.x;
    uint t = blockDim.x*blockIdx.x + tid;

    if(t < numConfs) {

        d_rmsds[t] = sqrt(calc_fused_lprmsd( numAtoms, numConfs, d_X, ctr_ID, t, G_, d_subset_flag_));

        // Error-testing: compare each conformer to itself.
        // This should return 0.   
        // d_rmsds[t] = calc_rmsd2_goodenough( numAtoms, numConfs, d_X, t, t, G_ );
    }

}
#endif
