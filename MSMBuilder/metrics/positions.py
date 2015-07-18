from __future__ import print_function, absolute_import, division
from mdtraj.utils.six import PY2

import logging
logger = logging.getLogger(__name__)
import mdtraj as md
import numpy as np

from .baseclasses import Vectorized, AbstractDistanceMetric

class Positions(Vectorized, AbstractDistanceMetric):

    """
    This metric will calculate distances based on some vector norm while
    doing only a single alignment to a target structure.

    This is NOT the RMSD since the structures are not pair-wise aligned,
    they are aligned to a single structure at the beginning.
    """
    allowable_scipy_metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                               'correlation', 'cosine', 'euclidean', 'minkowski',
                               'sqeuclidean', 'seuclidean', 'mahalanobis']

    def __init__(self, target, align_indices=None, atom_indices=None,
                 metric='euclidean', p=2):
        """Create a distance metric to act on absolute atom positions

        Parameters
        ----------
        target : mdtraj.Trajectory
            structure to align each conformation to
        align_indices : np.ndarray or None
            atom indices to use in the alignment step
        atom_indices : np.ndarray or None
            atom indices to use when calculating distances
        metric : {'braycurtis', 'canberra', 'chebyshev', 'cityblock',
                  'correlation', 'cosine', 'euclidean', 'minkowski',
                  'sqeuclidean', 'seuclidean', 'mahalanobis', 'sqmahalanobis'}
            Distance metric to equip the vector space with.
            or any combination thereof
        p : int, optional
            p-norm order, used for metric='minkowski'

        """
        s = super(Positions, self) if PY2 else super()
        s.__init__(metric, p)

        if not isinstance(target, md.Trajectory):
            raise ValueError("target must be mdtraj.Trajectory instance")

        if isinstance(align_indices, list) or isinstance(align_indices, np.ndarray):
            self.align_indices = np.array(align_indices).astype(int).flatten()
        else:
            self.align_indices = None

        if isinstance(atom_indices, list) or isinstance(atom_indices, np.ndarray):
            self.atom_indices = np.array(atom_indices).astype(int).flatten()
        else:
            self.atom_indices = None

        self.target = target
        self.target.center_coordinates()
        if self.align_indices is None:
            self._permute_groups = [[i] for i in xrange(self.target.n_atoms)]

        else:
            slf._permute_groups = [[i] for i in self.align_indices]


    def prepare_trajectory(self, trajectory, return_dist=False):
        """
        Prepare a trajectory by first aligning it to the target with LPRMSD
        then returning a reshaped array corresponding to the correct atom positions
        flattened into a vector

        Parameters:
        -----------
        trajectory : mdtraj.Trajectory instance
            trajectory to prepare
        return_dist : bool, optional
            this will align the frames in trajectory to self.target,
            if you want the aligned distances, then pass return_dist=True

        Returns
        -------
        prep_trajectory : np.ndarray
            prepared trajectory
        aligned_distances : np.ndarray
            only returned if return_dist==True. Distance to the target
            after alignment for each frame in trajectory
        """

        t = md.Trajectory(trajectory.xyz.copy(), topology=trajectory.topology.copy())

        distances = md.lprmsd(t, self.target, superpose=True, atom_indices=self.align_indices,
                              permute_groups=self._permute_groups)

        prep_trajectory = t.xyz[:, self.atom_indices].reshape((len(t), -1))

        return prep_trajectory
