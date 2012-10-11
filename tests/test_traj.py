import os, sys
import numpy as np
from msmbuilder.testing import *
from msmbuilder import Trajectory


def test_traj_0():
    
    aind = np.unique( np.random.randint( 22, size=4) )
    stride = np.random.randint(1, 100 )
    
    r_traj = get('Trajectories/trj0.lh5')

    r_traj.restrict_atom_indices( aind )

    r_traj['XYZList'] = r_traj['XYZList'][ ::stride ]

    traj = Trajectory.load_from_lhdf(get('Trajectories/trj0.lh5', just_filename=True),
        Stride=stride, AtomIndices=aind)

    for key in traj.keys():
        if key in ['SerializerFilename'] :
            continue
        
        if key in ['IndexList']:
            for row, r_row in zip( traj[key], r_traj[key] ):
                eq(row, r_row)
        elif key == 'XYZList':
            eq(traj[key], r_traj[key])
        else:
            eq(traj[key], r_traj[key])

def test_traj_1():
    for i in range(20):
        test_traj_0()
