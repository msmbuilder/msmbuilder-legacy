import os, sys
import numpy as np
import numpy.testing as npt
from nose.tools import ok_, eq_

from common import fixtures_dir

from msmbuilder import Trajectory


def test_traj_0():
    
    aind = np.unique( np.random.randint( 22, size=4) )
    stride = np.random.randint(1, 100 )
    
    r_traj = Trajectory.load_from_lhdf( os.path.join( fixtures_dir(), 'trj0.lh5' ), Stride=1 )

    r_traj.restrict_atom_indices( aind )

    r_traj['XYZList'] = r_traj['XYZList'][ ::stride ]

    traj = Trajectory.load_from_lhdf( os.path.join( fixtures_dir(), 'trj0.lh5' ), Stride = stride, AtomIndices = aind )

    for key in traj.keys():
        if key in ['SerializerFilename'] :
            continue
        
        if key in ['IndexList']:
            for row, r_row in zip( traj[key], r_traj[key] ):
                npt.assert_array_equal( row, r_row )
        elif key == 'XYZList':
            npt.assert_array_almost_equal( traj[key], r_traj[key])
        else:
            npt.assert_array_equal( traj[key], r_traj[key] )

def test_traj_1():
    for i in range(20):
        test_traj_0()
