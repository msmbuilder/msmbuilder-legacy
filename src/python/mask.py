#!/usr/bin/env python
import numpy, dihedral
import msmbuilder
from msmbuilder.io import *
from msmbuilder import arglib
import sys

def get_dihedral_mask(traj_path, traj_num, conf):
    '''Computes a mask for cis and trans states based on the K109/P110 dihedral angle for CheY.
    Can be modified to be for more general use later if needed.

    Arguments:
    - traj_path (str) : path to Trajectory directory with all trj#.lh5 files
    - traj_num (int) : the number of the trajectory currently being analyzed. This function
    assumes that the file name of the trajectory contains the number corresponding to the
    index of the trajectory in the Assingments.h5 file, as it is by default ,i.e. the first (0)
    entry in Assignments.h5 corresponds to trj0.lh5
    - conf (str) : Which states to isolate in mask and later when building count matrix.
    Choose either 'cis' or 'trans'.

    Returns:
    -mask (numpy array) : a 1-d array of 1's and 0's, with the 1's corresponding to the desired
    state you want to isolate (e.g. 'cis') for a frame of the trajectory, and 0 representing the
    other state which you don't want.
   '''
    indices = numpy.array([[1634, 1652, 1654, 1664]]) # Atom indices for K109/P110 omega angle
    traj_path = traj_path
    traj_name = 'trj%s.lh5'%(traj_num) # Assumes standard naming schemes for trajectory lh5 files
    traj = msmbuilder.Trajectory.load_trajectory_file(traj_path + '/%s'%(traj_name))
    omega = dihedral.compute_dihedrals(traj,indices)
    mask = numpy.zeros(len(traj)) # Length matches the number of frames in a trajectory
    for i in range(len(omega)):
        for j in omega[i]: # The omega values
            j = float(j)
            if conf == 'cis':
                if j <= 90:
                    mask[i] = 1 # Adds a 1 to the mask if cis
            elif conf == 'trans':
                if j > 90:
                    mask[i] = 1
            else:
                print 'Please either use "cis" or "trans" for the third argument'
                return 1
    return mask

def save_new_assignments(traj_path, input = 'Assignments.h5', output = 'Assignments_mod.h5', conf =
'cis'):
    '''Isolates all cis or trans states in the Assignments.h5 file by replacing all other states
    by -1's. When building the count matrix, these -1 entries will not be included. Writes a new
    assignments file with this new assignments matrix.

    Arguments:
    - traj_path (str) : path to Trajectory directory with all trj#.lh5 files
    - assignments (str) : path to file to modify. Default: Assignments.h5
    - output (str) : path to use for output assignments file. Default: Assignments_mod.h5
    - conf (str) : conformer to isolate and for which to build count matrix. Choose either 'cis' or
    'trans'. Default: 'cis'
    '''
    input_file = loadh(input)
    data = input_file['arr_0']
    for i in range(len(data)):
        traj_num = i
        mask = get_dihedral_mask(traj_path, traj_num, conf)
        for j in range(len(mask)):
            if mask[j] == 0:
                data[i][j] = -1
    output_file = saveh(output, arr_0 = data)
    return 0

if __name__ == '__main__':
    parser = arglib.ArgumentParser(description='''Uses an existing
    Assignments file (Default: Assignments.h5) for CheY and writes a new
    Assignments file (Default: Assignments_mod.h5) that isolates only the
    cis states, replacing all trans states with -1's.''')
    parser.add_argument('traj_path', type=str, help='''Path to the Trajectory
    directory with all the trj.lh5 data''')
    parser.add_argument('input', type=str, help='''Path to original Assignments
    file. Default: Assignments.h5''', default='Assignments.h5')
    parser.add_argument('output', type=str, help = '''Path to output new Assignments file.
    Default: Assignments_mod.h5''', default='Assignments_mod.h5')
    parser.add_argument('conf', type=str, help = '''Which conformer you want to isolate in the
    new Assignments file ('cis' or 'trans'). Default: cis''', default='cis')

    args = parser.parse_args()
    save_new_assignments(args.traj_path, args.input, args.output, args.conf)
