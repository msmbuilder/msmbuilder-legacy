import numpy as np
from msmbuilder.testing import *
from common import load_traj
from mdtraj.geometry import dihedral as _dihedralcalc

class TestDihedralCalc():
    """Test the msmbuilder.geometry.dihedral module"""
    
    def setUp(self):
        self.traj = load_traj()
        self.n_frames = self.traj['XYZList'].shape[0]
        self.n_atoms = self.traj['XYZList'].shape[1]
    
    def test_compute_dihedrals(self):
        indices = np.array([[0,1,2,3]])
        rads = _dihedralcalc.compute_dihedrals(self.traj, indices)
        
        eq(rads.shape, (3, 1))
        eq(rads.flatten(), np.array([0, 0, np.pi]))
        
        # No degrees any more
#         degrees = _dihedralcalc.compute_dihedrals(self.traj, indices, degrees=True)
#         eq(degrees.shape, (3,1))
#         eq(degrees.flatten(), np.array([0, 0, 180]))
        
    @skip('Not written')
    def test_get_indices(self):
        pass
