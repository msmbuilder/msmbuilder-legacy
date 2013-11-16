import logging
logger = logging.getLogger(__name__)
import numpy as np
from baseclasses import Vectorized
from msmbuilder import io

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
        _check_indices(solute_indices, 'Solute')
        _check_indices(solvent_indices, 'Solvent')

        super(SolventFp, self).__init__(metric, p, V, VI)
        self.solute_indices = solute_indices
        self.solvent_indices = solvent_indices
        self.sigma = sigma

    def __repr__(self):
        "String representation of the object"
        return ('metrics.SolventFp(metric=%s, p=%s, sigma=%s)'
                % (self.metric, self.p, self.sigma))

    def prepare_trajectory(self, trajectory):
        """lalalala
        Parameters
        ----------
        trajectory : msmbuilder.Trajectory
            An MSMBuilder trajectory to prepare

        Returns
        -------
        lalal : ndarray
            A 2D array of lalala
        """

        # Give shorter names to these things
        prot_indices = self.solute_indices
        water_indices = self.solvent_indices
        sigma = self.sigma
        traj = trajectory['XYZList']

        # The result vector
        fingerprints = np.zeros((len(traj), len(prot_indices)))
        
        for frame_t in xrange(len(traj)):
            for i, prot_i in enumerate(prot_indices):
                for water_j in water_indices:
                    fingerprints[frame_t, i] += _kernel(traj[frame_t][prot_i], traj[frame_t][water_j], sigma)

        return fingerprints

def _kernel(x, y, sigma):
    """Gaussian kernel K(x, y)."""
    diff = x - y
    dot = np.dot(diff, diff)
    return np.exp(-dot / (2.0 * sigma * sigma))

def _check_indices(indices, name):
    """Validate input indices."""
    if indices is not None:
        if not isinstance(indices, np.ndarray):
            raise ValueError('%s indices must be a numpy array' % name)
        if not indices.ndim == 1:
            raise ValueError('%s indices must be 1D' % name)
        if not indices.dtype == np.int:
            raise ValueError('%s indices must contain ints' % name)
    else:
        raise ValueError('%s indices must be specified' % name)

class OuterProductAssignment(object):
    """Class to facilitate taking the outer product of the result of
    two clusterings."""

    def __init__(self, ass1_fn, ass2_fn):
        """Create the object

        the ass_fn's should point to an hdf5 assignments file.
        """
        self.ass1 = io.loadh(ass1_fn, 'arr_0')
        self.ass2 = io.loadh(ass2_fn, 'arr_0')

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

                new_ass[i, j] = translations[self.ass1[i, j], self.ass2[i, j]]

        return new_ass














