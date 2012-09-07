#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


static inline float euclidean3(const float* v1, const float* v2) {
    float d0 = v1[0] - v2[0];
    float d1 = v1[1] - v2[1];
    float d2 = v1[2] - v2[2];   
    return sqrt(d0*d0 + d1*d1 + d2*d2);
}

void asa_frame(const float* frame, const int n_atoms, const float* atom_radii,
               const float* sphere_points, const int n_sphere_points,
               int* neighbor_indices, float* centered_sphere_ponts,
               float* areas) {
    // Calculate the accessible surface area of each atom in a single snapshot
    // 
    // Parameters
    // ----------
    // frame : 2d array, shape=[n_atoms, 3]
    //     The coordinates of the nuclei
    // n_atoms : int
    //     the major axis length of frame
    // atom_radii : 1d array, shape=[n_atoms]
    //     the van der waals radii of the atoms PLUS the probe radius
    // sphere_points : 2d array, shape=[n_sphere_points, 3]
    //     a bunch of uniformly distributed points on a sphere
    // n_sphere_points : int
    //    the number of sphere points
    
    // centered_sphere_ponts : WORK BUFFER 2d array, shape=[n_sphere_points, 3]
    //    empty memory that intermediate calculations can be stored in
    // neighbor_indices : WORK BUFFER 2d array, shape=[n_atoms]
    //    empty memory that intermediate calculations can be stored in
    // NOTE: the point of these work buffers is that if we want to call
    //    this function repreatedly, its more efficient not to keep re-mallocing
    //    these work buffers, but instead just reuse them.
    
    // areas : 1d array, shape=[n_atoms]
    //     the output buffer to place the results in -- the surface area of each
    //     atom
    
    int i, j, k, k_prime;
    int n_neighbor_indices, is_accessible, k_closest_neighbor;
    float r;
    float constant = 4.0 * M_PI / n_sphere_points;
    
    for (i = 0; i < n_atoms; i++) {
        // Get all the atoms close to atom `i`
        n_neighbor_indices = 0;
        for (j = 0; j < n_atoms; j++) {
            if (i == j) {
              continue;
            }
            
            r = euclidean3(frame+i*3, frame+j*3);
            if (r < atom_radii[i] + atom_radii[j]) {
                neighbor_indices[n_neighbor_indices]  = j;
                n_neighbor_indices++;
            }
            
            if (r < 1.9e-5) {
              printf("ERROR: THIS CODE IS KNOWN TO FAIL WHEN ATOMS ARE VIRTUALLY");
              printf("ON TOP OF ONE ANOTHER. YOU SUPPLIED TWO ATOMS %f", r);
              printf("APART. QUITTING NOW");
              exit(1);
            }
        }
        
        // Center the sphere points on atom i
        for (j = 0; j < n_sphere_points; j++) {
            centered_sphere_ponts[3*j + 0] = frame[3*i + 0] + (sphere_points[3*j + 0] * atom_radii[i]);
            centered_sphere_ponts[3*j + 1] = frame[3*i + 1] + (sphere_points[3*j + 1] * atom_radii[i]);
            centered_sphere_ponts[3*j + 2] = frame[3*i + 2] + (sphere_points[3*j + 2] * atom_radii[i]);
        }
        
        // Check if each of these points is accessible
        k_closest_neighbor = 0;
        for (j = 0; j < n_sphere_points; j++) {
            is_accessible = 1;
            
            // iterate through the sphere points by cycling through them
            // in a circle, starting with k_closest_neighbor and then wrapping
            // around
            for (k = k_closest_neighbor; k < n_neighbor_indices + k_closest_neighbor; k++) {
                k_prime = k % n_neighbor_indices;
                r = atom_radii[neighbor_indices[k_prime]];
                
                if (euclidean3(centered_sphere_ponts+3*j, frame+3*neighbor_indices[k_prime]) < r) {
                    k_closest_neighbor = k;
                    is_accessible = 0;
                    break;
                }
            }
            
            if (is_accessible) {
                areas[i]++;
            }
        }
        
        areas[i] *= constant * (atom_radii[i])*(atom_radii[i]);
    }
}


void generate_sphere_points(float* sphere_points, int n_points) {
    // Compute the coordinates of points on a sphere using the
    // Golden Section Spiral algorithm.
    // 
    // Parameters
    // ----------
    // sphere_points : float*
    //     Empty array of length n_points*3 -- will be filled with the points
    //     as an array in C-order. i.e. sphere_points[3*i], sphere_points[3*i+1]
    //     and sphere_points[3*i+2] are the x,y,z coordinates of the ith point
    // n_pts : int
    //     Number of points to generate on the sphere
    // 
    // """
    int i;
    float y, r, phi;
    float inc = M_PI * (3.0 - sqrt(5));
    float offset = 2.0 / n_points;
    
    for (i = 0; i < n_points; i++) {
        y = i * offset - 1.0 + (offset / 2.0);
        r = sqrt(1.0 - y*y);
        phi = i * inc;
        
        sphere_points[3*i] = cos(phi) * r;
        sphere_points[3*i+1] = y;
        sphere_points[3*i+2] = sin(phi) * r;
    }
}


void asa_trajectory(const int n_frames, const int n_atoms, const float* xyzlist,
		    const float* atom_radii, const int n_sphere_points, float* array_of_areas){
    // Calculate the accessible surface area of each atom in each frame of
    // a trajectory
    // 
    // Parameters
    // ----------
    // xyzlist : 3d array, shape=[n_frames, n_atoms, 3]
    //     The coordinates of the nuclei
    // n_frames : int
    //     the number of frames in the trajectory
    // n_atoms : int
    //     the number of atoms in each frame
    // atom_radii : 1d array, shape=[n_atoms]
    //     the van der waals radii of the atoms PLUS the probe radius
    // n_sphere_points : int
    //     number of points to generate sampling the unit sphere. higher is
    //     better (more accurate) but more expensive
    // array_of_areas : 2d array, shape=[n_frames, n_atoms]
    //     the output buffer to place the results in -- the surface area of each
    //     frame (each atom within that frame)
    
    int i;

    //work buffers that will be thread-local
    int* wb1;
    float* wb2;
    
    //generate the sphere points
    float* sphere_points = (const float*) malloc(n_sphere_points*3*sizeof(float));
    generate_sphere_points(sphere_points, n_sphere_points);
    
    #pragma omp parallel private(wb1, wb2)
    {
        // malloc the work buffers for each thread
        wb1 = (int*) malloc(n_atoms*sizeof(int));
        wb2 = (float*) malloc(3*n_sphere_points*sizeof(float));

        #pragma omp for
        for (i = 0; i < n_frames; i++) {
            asa_frame(xyzlist + i*n_atoms*3, n_atoms, atom_radii, sphere_points, 
                n_sphere_points, wb1, wb2, array_of_areas + i*n_atoms);
        }
        
        free(wb1);
        free(wb2);
    }
    
    free(sphere_points);
}
