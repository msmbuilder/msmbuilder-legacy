#ifndef _CPY_DISTANCE_H
#define _CPY_DISTANCE_H

void dihedrals_from_traj(double *results, const double *xyzlist, const long *quartets, int traj_length, int num_atoms, int num_quartets);
void dihedrals_from_traj_float(float *results, const float *xyzlist, const long *quartets, int traj_length, int num_atoms, int num_quartets);

#endif
