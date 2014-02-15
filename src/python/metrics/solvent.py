import logging

import mdtraj as md

from baseclasses import Vectorized
import numpy as np


logger = logging.getLogger(__name__)


class SolventFp(Vectorized):
    """Distance metric for calculating distances between frames based on their
    solvent signature as in Gu et al. BMC Bioinformatics 2013, 14(Suppl 2):S8.
    """

    allowable_scipy_metrics = ['braycurtis', 'canberra', 'chebyshev',
                               'cityblock', 'correlation', 'cosine',
                               'euclidean', 'minkowski', 'sqeuclidean',
                               'seuclidean', 'mahalanobis', 'sqmahalanobis']

    def __init__(self, solute_indices, solvent_indices, sigma,
                 metric='euclidean', p=2, V=None, VI=None):
        """Create a distance metric to capture solvent degrees of freedom

        Parameters
        ----------
        solute_indices : ndarray
            atom indices of the solute atoms
        solvent_indices : ndarray
            atom indices of the solvent atoms
        sigma : float
                width of gaussian kernel
        metric : {'braycurtis', 'canberra', 'chebyshev', 'cityblock',
                  'correlation', 'cosine', 'euclidean', 'minkowski',
                  'sqeuclidean', 'seuclidean', 'mahalanobis', 'sqmahalanobis'}
            Distance metric to equip the vector space with.
        p : int, optional
            p-norm order, used for metric='minkowski'
        V : ndarray, optional
            variances, used for metric='seuclidean'
        VI : ndarray, optional
            inverse covariance matrix, used for metric='mahalanobi'

        """

        # Check input indices
        md.utils.ensure_type(solute_indices, dtype=np.int, ndim=1,
                             name='solute', can_be_none=False)
        md.utils.ensure_type(solvent_indices, dtype=np.int, ndim=1,
                             name='solvent', can_be_none=False)

        super(SolventFp, self).__init__(metric, p, V, VI)
        self.solute_indices = solute_indices
        self.solvent_indices = solvent_indices
        self.sigma = sigma

    def __repr__(self):
        "String representation of the object"
        return ('metrics.SolventFp(metric=%s, p=%s, sigma=%s)'
                % (self.metric, self.p, self.sigma))

    def prepare_trajectory(self, trajectory):
        """Calculate solvent fingerprints
        Parameters
        ----------
        trajectory : msmbuilder.Trajectory
            An MSMBuilder trajectory to prepare

        Returns
        -------
        fingerprints : ndarray
            A 2D array of fingerprint vectors of
            shape (traj_length, protein_atom)
        """

        # Give shorter names to these things
        prot_indices = self.solute_indices
        water_indices = self.solvent_indices
        sigma = self.sigma

        # The result vector
        fingerprints = np.zeros((trajectory.n_frames, len(prot_indices)))

        # Check for periodic information
        if trajectory.unitcell_lengths is None:
            logging.warn('No periodic information found for computing solventfp.')

        for i, prot_i in enumerate(prot_indices):
            # For each protein atom, calculate distance to all water
            # molecules
            atom_pairs = np.empty((len(water_indices), 2))
            atom_pairs[:, 0] = prot_i
            atom_pairs[:, 1] = water_indices
            # Get a traj_length x n_water_indices vector of distances
            distances = md.compute_distances(trajectory,
                                             atom_pairs,
                                             periodic=True)
            # Calculate guassian kernel
            distances = np.exp(-distances / (2 * sigma * sigma))

            # Sum over water atoms for all frames
            fingerprints[:, i] = np.sum(distances, axis=1)

        return fingerprints


class OuterProductAssignment(object):
    """Class to facilitate taking the outer product of the result of
    two clusterings."""

    def __init__(self, assign1, assign2):
        """Create a distance metric to capture solvent degrees of freedom

        Parameters
        ----------
        assign1 : ndarray
            assignment array 1
        assign2 : ndarray
            assignment array 2
        """
        self.ass1 = assign1
        self.ass2 = assign2

    def get_product_assignments(self):
        assert self.ass1.shape == self.ass2.shape, """Assignments must be
            for the same set of trajectories."""
        new_ass = -1 * np.ones_like(self.ass1, dtype=np.int)

        nstates1 = np.max(self.ass1) + 1
        nstates2 = np.max(self.ass2) + 1

        translations = np.reshape(np.arange(nstates1 * nstates2),
                                  (nstates1, nstates2))

        ass_shape = self.ass1.shape
        for i in xrange(ass_shape[0]):
            for j in xrange(ass_shape[1]):
                if self.ass1[i, j] == -1:
                    # No assignment here
                    assert self.ass2[i, j] == -1, """Assignments must be for
                        the same set of trajectories."""
                else:
                    new_ass[i, j] = translations[self.ass1[i, j], self.ass2[i, j]]

        return new_ass














