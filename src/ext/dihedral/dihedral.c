#include <stdio.h>
#include <math.h>

inline void crossProduct3(double a[], const double b[], const double c[]) {
    //Calculate the cross product between length-three vectors b and c, storing
    //the result in a
    (a)[0] = (b)[1] * (c)[2] - (c)[1] * (b)[2];
    (a)[1] = (b)[2] * (c)[0] - (c)[2] * (b)[0];
    (a)[2] = (b)[0] * (c)[1] - (c)[0] * (b)[1];
}

inline double dotProduct3(const double b[], const double c[]) {
    //Calculate the dot product between length-three vectors b and c
    return b[0] * c[0] + b[1] * c[1] + b[2] * c[2];
}

double dihedral(const double *x0, const double *x1, const double *x2,
                const double *x3) {
    //Calculate the signed dihedral angle between four points.
    //Result in radians
    //x0, x1, x2, x3 should be length three arrays
    int i;
    double b1[3], b2[3], b3[3], c1[3], c2[3];
    double arg1, arg2, b2_norm;
    
    for (i = 0; i < 3; i++) {
        b1[i] = x1[i] - x0[i];
        b2[i] = x2[i] - x1[i];
        b3[i] = x3[i] - x2[i];
    }
    
    crossProduct3(c1, b2, b3);
    crossProduct3(c2, b1, b2);
    
    arg1 = dotProduct3(b1, c1);
    b2_norm = sqrt(dotProduct3(b2, b2));
    arg1 = arg1 * b2_norm;

    arg2 = dotProduct3(c2, c1);
    return atan2(arg1, arg2);
}

void dihedrals_from_traj(double *results, const double *xyzlist, const long *quartets,
                         int traj_length, int num_atoms, int num_quartets) {
    // results is a 2D array (traj_length x num_quartets). xyzlist is a 3D
    // array (traj_length x num_atoms x 3). quartets is a 2D array (num_quartets
    // x 4) where each row is the four atomindices of the atoms to calculate the
    // dihedral angle between
    // results are storted in the results array
    int i,j,k;
    long e[4] = {0,0,0,0};
    double *x[4] = {0,0,0,0};
    double *result_ptr;
    
    #pragma omp parallel for default(none) shared(results, xyzlist, quartets, traj_length, num_atoms, num_quartets) private(j, k, e, x, result_ptr)
    for (i = 0; i < traj_length; i++) {
        for (j = 0; j < num_quartets; j++) {
            for (k = 0; k < 4; k++) {
                e[k] = quartets[4*j + k];
                x[k] = xyzlist + i*num_atoms*3 + e[k]*3;
            }
            /*printf("i: %d -- j %d\n", i, j);
            printf("first %f", *(xyzlist + i * traj_length*3 + 4*3));
            printf("e -- %d %d %d %d\n", e[0], e[1], e[2], e[3]);
            printf("x0: %f\n", *(x[0]));
            printf("x1: %f\n", *(x[1]));
            printf("x2: %f\n", *(x[2]));
            printf("Calculated %f\n", dihedral(x[0], x[1], x[2], x[3]));*/
            
            result_ptr = results + i * num_quartets + j;
            *result_ptr = dihedral(x[0], x[1], x[2], x[3]);
        }
    }
}

/* 

Another version of the code that uses floats instead of doubles

*/

inline void crossProduct3_float(float a[], const float b[], const float c[]) {
    //Calculate the cross product between length-three vectors b and c, storing
    //the result in a
    (a)[0] = (b)[1] * (c)[2] - (c)[1] * (b)[2];
    (a)[1] = (b)[2] * (c)[0] - (c)[2] * (b)[0];
    (a)[2] = (b)[0] * (c)[1] - (c)[0] * (b)[1];
}

inline double dotProduct3_float(const float b[], const float c[]) {
    //Calculate the dot product between length-three vectors b and c
    return b[0] * c[0] + b[1] * c[1] + b[2] * c[2];
}

double dihedral_float(const float *x0, const float *x1, const float *x2,
                const float *x3) {
    //Calculate the signed dihedral angle between four points.
    //Result in radians
    //x0, x1, x2, x3 should be length three arrays
    int i;
    float b1[3], b2[3], b3[3], c1[3], c2[3];
    float arg1, arg2, b2_norm;
    
    for (i = 0; i < 3; i++) {
        b1[i] = x1[i] - x0[i];
        b2[i] = x2[i] - x1[i];
        b3[i] = x3[i] - x2[i];
    }
    
    crossProduct3_float(c1, b2, b3);
    crossProduct3_float(c2, b1, b2);
    
    arg1 = dotProduct3_float(b1, c1);
    b2_norm = sqrt(dotProduct3_float(b2, b2));
    arg1 = arg1 * b2_norm;

    arg2 = dotProduct3_float(c2, c1);
    return atan2(arg1, arg2);
}

void dihedrals_from_traj_float(float *results, const float *xyzlist, const long *quartets,
                         int traj_length, int num_atoms, int num_quartets) {
    // results is a 2D array (traj_length x num_quartets). xyzlist is a 3D
    // array (traj_length x num_atoms x 3). quartets is a 2D array (num_quartets
    // x 4) where each row is the four atomindices of the atoms to calculate the
    // dihedral angle between
    // results are storted in the results array
    int i,j,k;
    long e[4] = {0,0,0,0};
    float *x[4] = {0,0,0,0};
    float *result_ptr;
    
    #pragma omp parallel for default(none) shared(results, xyzlist, quartets, traj_length, num_atoms, num_quartets) private(j, k, e, x, result_ptr)
    for (i = 0; i < traj_length; i++) {
        for (j = 0; j < num_quartets; j++) {
            for (k = 0; k < 4; k++) {
                e[k] = quartets[4*j + k];
                x[k] = xyzlist + i*num_atoms*3 + e[k]*3;
            }
            /*printf("i: %d -- j %d\n", i, j);
            printf("first %f", *(xyzlist + i * traj_length*3 + 4*3));
            printf("e -- %d %d %d %d\n", e[0], e[1], e[2], e[3]);
            printf("x0: %f\n", *(x[0]));
            printf("x1: %f\n", *(x[1]));
            printf("x2: %f\n", *(x[2]));
            printf("Calculated %f\n", dihedral(x[0], x[1], x[2], x[3]));*/
            
            result_ptr = results + i * num_quartets + j;
            *result_ptr = dihedral_float(x[0], x[1], x[2], x[3]);
        }
    }
}