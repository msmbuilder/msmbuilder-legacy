import sys, os
import numpy as np
import hashlib
import IPython as ip
from msmbuilder.metrics import AbstractDistanceMetric
from msmbuilder.clustering import concatenate_trajectories
from msmbuilder import Project
import argparse
import logging
logger = logging.getLogger('gpurmsd')
from swGPURMSD import RMSD as _GPURMSD

description = """Please cite

Zhao, Yutong; Sheong, Fu-Kit, Sun, Jian; Huang, Xuhui, "A Fast, GPU-powered, Clustering Algorithm of Conformations" In Press.
"""

class GPURMSD(AbstractDistanceMetric):
    name = 'GPURMSD'

    @classmethod
    def from_args(cls, args):
        atomindices = np.loadtxt(args.gpurmsd_atom_indices, dtype=np.int)
        return cls(atomindices)

    def __init__(self, atomindices=None):
        self.atomindices = atomindices
        self._gpurmsd = None

    def prepare_trajectory(self, trajectory):
        logger.info('GPU preparing')
        n_confs = len(trajectory)

        if self._gpurmsd != None:
            raise ValueError("messed up call pattern")

        xyzlist = trajectory['XYZList']
        if self.atomindices != None:
            xyzlist = xyzlist[:, self.atomindices, :]

        xyzlist = np.array(xyzlist.swapaxes(1,2).swapaxes(0,2), copy=True, order='C')        
        self._gpurmsd = _GPURMSD(xyzlist)
        self._gpurmsd.set_subset_flag_array(np.ones(n_confs, dtype=np.int32))
        self._gpurmsd.center_and_precompute_G()

        return np.arange(n_confs)

    def one_to_all(self, ptraj1, ptraj2, index1):
        """Compute the distance from the `index1th` frame to the rest of the
        frames

        Note, for GPURMSD, ptraj1 and ptraj2 must be the same, and 
        must be the trajectories that were passed to __init__ to be loaded
        onto the GPU
        """
        logger.info('GPU computing')
        assert np.count_nonzero(ptraj1 - ptraj2) == 0

        results = np.empty(len(ptraj1), dtype=np.float32)
        self._gpurmsd.set_rmsd_array(results)

        #logger.info('GPU %s', type(index1))

        self._gpurmsd.all_against_one_rmsd(int(index1))

        return results
    

def add_metric_parser(parsergroup, add_argument):
    logger.info('GPU adding parser')
    gpurmsd = parsergroup.add_parser('GPURMSD', description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_argument(gpurmsd, '-a', dest='gpurmsd_atom_indices', default='AtomIndices.dat')
    return gpurmsd

        
