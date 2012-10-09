import numpy as np
import numpy.testing as npt


from msmbuilder.geometry import dihedral as _dihedralcalc
from common import load_traj, skip

class TestDihedralCalc():
    """Test the msmbuilder.geometry.dihedral module"""
    
    def setUp(self):
        self.traj = load_traj()
        self.n_frames = self.traj['XYZList'].shape[0]
        self.n_atoms = self.traj['XYZList'].shape[1]
    
    def test_compute_dihedrals(self):
        indices = np.array([[0,1,2,3]])
        rads = _dihedralcalc.compute_dihedrals(self.traj, indices, degrees=False)
        
        npt.assert_array_equal(rads.shape, [3,1])
        npt.assert_array_almost_equal(rads.flatten(), [0, 0, np.pi])
        
        degrees = _dihedralcalc.compute_dihedrals(self.traj, indices, degrees=True)
        npt.assert_array_equal(degrees.shape, [3,1])
        npt.assert_array_almost_equal(degrees.flatten(), [0, 0, 180])
        
    @skip('Not written')
    def test_get_indices(self):
        pass
