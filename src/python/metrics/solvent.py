import logging
logger = logging.getLogger(__name__)
import numpy as np
from baseclasses import Vectorized

class SolventFp(Vectorized):
    """Distance metric for calculating distances between frames based on their
    solvent signature as in Gu et al. BMC Bioinformatics 2013, 14(Suppl 2):S8.
    """

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
        fingerprints = np.zeros(len(prot_indices))

        for i, prot_i in enumerate(prot_indices):
            for water_j in water_indices:
                fingerprints[i] += _kernel(traj[prot_i], traj[water_j], sigma)

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


