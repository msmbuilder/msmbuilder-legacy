from itertools import product
import numpy as np
import numpy.testing as npt

from msmbuilder.geometry import contact as _contactcalc
from common import load_traj

class TestContactCalc():
    """Test the msmbuilder.geometry.contact module"""
    
    def setUp(self):
        self.traj = load_traj()
        self.n_frames = self.traj['XYZList'].shape[0]
        self.n_atoms = self.traj['XYZList'].shape[1]
    
    def test_atom_distances(self):
        pairs = np.array(list(product(xrange(self.n_atoms), repeat=2)))
        distances = _contactcalc.atom_distances(self.traj['XYZList'], pairs)
        
        same_atom = np.where(pairs[:,0] == pairs[:,1])        
        flipped_pairs = []
        
        for i in xrange(pairs.shape[0]):
            for j in xrange(pairs.shape[0]):
                if (pairs[i,0] == pairs[j,1]) and (pairs[i,1] == pairs[j,0]) \
                    and (i != j):
                    flipped_pairs.append((i,j))
        flipped_pairs = np.array(flipped_pairs)
        
        for i in xrange(self.n_frames):
            npt.assert_array_equal(distances[i,flipped_pairs[:,0]], distances[i, flipped_pairs[:,1]])
            npt.assert_array_equal(distances[i, same_atom], np.zeros_like(same_atom))
        
        
        # pair[1] is the distance from atom 0 to atom 1, which should be sqrt(3)
        # in the first frame (1,1,1) to (2,2,2)
        npt.assert_array_equal(pairs[1], [0,1])
        npt.assert_almost_equal(distances[0, 1], np.sqrt(3))
