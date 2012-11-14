#include <math.h>

void rg(const float *xyzlist, int traj_length, int num_atoms, double* results) {
    int i, j;
    double I;
    double meanx, meany, meanz;
    const float *frame;
    double x, y, z;
    double *results_ptr;
    
    #pragma omp parallel for default(none) shared(results, xyzlist, traj_length, num_atoms) private(j, I, meanx, meany, meanz, frame, x, y, z, results_ptr)
    for (i = 0; i < traj_length; i++) {
        results_ptr = results + i;
        frame = xyzlist + num_atoms * 3 * i;
        // find mean x, mean y, mean z
        meanx = 0;
        meany = 0;
        meanz = 0;
        for (j = 0; j < num_atoms; j++) {
            //printf("weights[%d], %f\n", j, weights[j]);
            //printf("x coord %f\n", *(frame + j *3));
            meanx += *(frame + j * 3);
            meany += *(frame + j * 3 + 1);
            meanz += *(frame + j * 3 + 2);
        }
        
        meanx /= num_atoms;
        meany /= num_atoms;
        meanz /= num_atoms;
        
        //printf("Frame %d\n", i);
        //printf("Meanx: %f\n", meanx);
        //printf("Meany: %f\n", meany);
        //printf("Meanz: %f\n\n", meanz);
        
        I = 0.0;
        for (j = 0; j < num_atoms; j++) {
            x = *(frame + j * 3);
            y = *(frame + j * 3 + 1);
            z = *(frame + j * 3 + 2);
            I += ((x-meanx)*(x-meanx) + (y-meany)*(y-meany) + (z-meanz)*(z-meanz));
        }
        I /= (num_atoms);
        
        //printf("I: %f", I);
        *results_ptr = sqrt(I);
    }
}

