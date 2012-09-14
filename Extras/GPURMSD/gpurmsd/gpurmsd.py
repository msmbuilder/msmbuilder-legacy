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
from Xrange import Xrange

description = """Please cite

Zhao, Yutong; Sheong, Fu-Kit, Sun, Jian; Huang, Xuhui, "A Fast, GPU-powered, Clustering Algorithm of Conformations" In Press.
"""

class GPURMSD(AbstractDistanceMetric):
    def __init__(self, atomindices=None):
        self.atomindices = atomindices
        self._gpurmsd = None

    def prepare_trajectory(self, trajectory):
        """Prepare a trajectory for disance calculation on the GPU.

        Parameters
        ----------
        trajectory : msmbuilder trajectory
            The trajectory to send to the GPU
        
        Returns
        -------
        prepared_trajectory : np.ndarray
            Don't worry about what the return value is :) its just input
            that you should give to GPURMSD.one_to_all().

        Notes
        -----
        - the GPU kernels except the xyz coordinates to be layed out
          as n_atoms x n_dimensions x n_frames, which means we need to
          first reorder them on the GPU before sending them.
        - You can only run prepare_trajectory once -- you have to send
          all the frames to the GPU in one go.
        """
        logger.info('GPU preparing')

        if self._gpurmsd != None:
            raise ValueError("messed up call pattern")

        xyzlist = trajectory['XYZList']
        n_confs, n_atoms = xyzlist.shape[0:2]

        if self.atomindices != None:
            xyzlist = xyzlist[:, self.atomindices, :]
            n_atoms = len(self.atomindices)

        xyzlist = np.array(xyzlist.swapaxes(1,2).swapaxes(0,2), copy=True, order='C')        
        self._gpurmsd = _GPURMSD(xyzlist)

        self._gpurmsd.set_subset_flag_array(np.ones(n_atoms, dtype=np.int32))
        self._gpurmsd.center_and_precompute_G()
        self._n_confs = n_confs

        return Xrange(self._n_confs)

    def one_to_all(self, ptraj1, ptraj2, index1):
        """Compute the distance from the `index1th` frame in ptraj1 to
        the rest of the frames in ptraj2

        """

        # should do some sanity checking to ensure that
        # ptraj1 and ptraj2 are really associated with
        # self._gpurmsd. Could "annotate" the xrange returned
        # by prepare_trajectory and then check for its watermark
        # here

        # the problem is that ptraj can either be a ndarray or an Xrange

        #logger.info('GPU computing')

        results = np.empty(self._n_confs, dtype=np.float32)
        self._gpurmsd.all_against_one_rmsd(int(index1))
        self._gpurmsd.retrieve_rmsds_from_device(results)

        if isinstance(ptraj2, Xrange):
            # get its slice representation
            return results[ptraj2.to_slice()]
        return results[ptraj2]
    

def add_metric_parser(parsergroup, add_argument):
    logger.info('GPU adding parser')
    gpurmsd = parsergroup.add_parser('GPURMSD', description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_argument(gpurmsd, '-a', dest='gpurmsd_atom_indices', default='AtomIndices.dat')
    return gpurmsd


def construct_gpurmsd(args):
    if args.metric != 'GPURMSD':
        return None

    atomindices = np.loadtxt(args.gpurmsd_atom_indices, dtype=np.int)
    return GPURMSD(atomindices)
