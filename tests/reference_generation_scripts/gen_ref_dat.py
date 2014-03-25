from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import sys, re, os


def get_data( fn ):

    data_file = open( fn, 'r' )

    lines = [ l for l in data_file.readlines() if not l[0] in ['#','@'] ]
    dat = np.genfromtxt(lines)

    return dat[:,1]

os.mkdir('Atoms')


for frame_ind in range(501):
    os.system('echo "0 0 " | g_sas -ndots {ndots} -f {xtc_fn} -s {pdb_fn} -b {i} -e {i} -oa Atoms/atom{i}.xvg -o Atoms/tot{i}.xvg'.format( xtc_fn='./trj0.xtc', pdb_fn='./native.pdb', i=frame_ind, ndots=960 ) )

filenames = [ 'Atoms/atom%d.xvg'%i for i in range(501) ]

traj_dat = []

for i in range(501):
    traj_dat.append( get_data( filenames[i] ) )

traj_dat = np.array( traj_dat )

np.savetxt('ref_dat.dat', traj_dat )
