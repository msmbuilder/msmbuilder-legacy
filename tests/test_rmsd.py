from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys, os
import numpy as np
import numpy.testing as npt
import scipy.spatial.distance
from unittest import skipIf
from mdtraj.utils.six.moves import xrange

from msmbuilder import metrics
from common import load_traj

class TestRMSD():
    "Test the msmbuilder.metrics.RMSD module"
    
    def setup(self):
        self.traj = load_traj()
        self.n_frames = self.traj.n_frames
        self.n_atoms = self.traj.n_atoms
        
        # RMSD from frame 0 to other frames
        self.target = np.array([0,0,0.63297522])
    
    def test_prepare(self):
        rmsds = [metrics.RMSD(), # all atom indices
                metrics.RMSD(range(self.n_atoms)),
                metrics.RMSD(xrange(self.n_atoms)),
                metrics.RMSD(np.arange(self.n_atoms))]
       
        for metric in rmsds:
            ptraj = metric.prepare_trajectory(self.traj)
    
    def test_one_to_all(self):
        for rmsd in [metrics.RMSD(), metrics.RMSD(omp_parallel=False)]:
            ptraj = rmsd.prepare_trajectory(self.traj)
            d0 = rmsd.one_to_all(ptraj, ptraj, 0)
        
            npt.assert_array_almost_equal(d0, self.target)

    def test_one_to_many(self):
        for rmsd in [metrics.RMSD(), metrics.RMSD(omp_parallel=False)]:
            ptraj = rmsd.prepare_trajectory(self.traj)
            for i in range(self.n_frames):
                di = rmsd.one_to_many(ptraj, ptraj, 0, [i])
                npt.assert_approx_equal(self.target[i], di)
    
    def test_all_pairwise(self):
        sys.stderr = open('/dev/null')
        for rmsd in [metrics.RMSD(), metrics.RMSD(omp_parallel=False)]:
            ptraj = rmsd.prepare_trajectory(self.traj)
            d1 = rmsd.all_pairwise(ptraj)
            target = [ 0., 0.63297522,  0.63297522]
            npt.assert_array_almost_equal(d1, target)

        sys.stderr=sys.__stderr__
