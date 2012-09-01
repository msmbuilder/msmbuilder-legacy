#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


inline float euclidean3(const float* v1, const float* v2) {
    return sqrt((v1[0] - v2[0])*(v1[0] - v2[0]) + 
                (v1[1] - v2[1])*(v1[1] - v2[1]) +
                (v1[2] - v2[2])*(v1[2] - v2[2]));
}


void asa_frame(const float* frame, const int n_atoms, const float* atom_radii, const float* sphere_points,
               const int n_sphere_points, float* areas) {
    int i, j, k, k_prime;
    int n_neighbor_indices, is_accessible, k_closest_neighbor;
    float r;
    float constant = 4.0 * M_PI / n_sphere_points;
    
    int* neighbor_indices = (int*) malloc(n_atoms*sizeof(int));
    float* centered_sphere_ponts = (float*) malloc(3*n_sphere_points*sizeof(float));
    
    
    for (i = 0; i < n_atoms; i++) {
        
        // Get all the atoms close to atom `i`
        n_neighbor_indices = 0;
        for (j = 0; j < n_atoms; j++) {
            if (euclidean3(frame+i*3, frame+j*3) < atom_radii[i] + atom_radii[j]) {
                neighbor_indices[n_neighbor_indices]  = j;
                n_neighbor_indices++;
            }
        }
        
        // Center the sphere points on atom i
        for (j = 0; j < n_sphere_points; j++) {
            centered_sphere_ponts[3*j + 0] = frame[3*i + 0] + (sphere_points[3*j + 0] * atom_radii[i]);
            centered_sphere_ponts[3*j + 1] = frame[3*i + 1] + (sphere_points[3*j + 1] * atom_radii[i]);
            centered_sphere_ponts[3*j + 2] = frame[3*i + 2] + (sphere_points[3*j + 2] * atom_radii[i]);
        }
        
        // Check if each of these points is accessible
        int k_closest_neighbor = 0;
        for (j = 0; j < n_sphere_points; j++) {
            is_accessible = 1;
            
            // iterate through the sphere points
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
    free(neighbor_indices);
    free(centered_sphere_ponts);
}