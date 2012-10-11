import logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger('metrics')
import numpy as np
from baseclasses import Vectorized, AbstractDistanceMetric
from msmbuilder.geometry import dihedral as _dihedralcalc

class Dihedral(Vectorized, AbstractDistanceMetric):
    """Distance metric for calculating distances between frames based on their
    projection in dihedral space."""
    
    allowable_scipy_metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                               'correlation', 'cosine', 'euclidean', 'minkowski',
                               'sqeuclidean', 'seuclidean', 'mahalanobis', 'sqmahalanobis']
    
    def __init__(self, metric='euclidean', p=2, angles='phi/psi', V=None, VI=None,
        indices=None):
        """Create a distance metric to act on torison angles
        
        Parameters
        ----------
        metric : {'braycurtis', 'canberra', 'chebyshev', 'cityblock',
                  'correlation', 'cosine', 'euclidean', 'minkowski',
                  'sqeuclidean', 'seuclidean', 'mahalanobis', 'sqmahalanobis'}
            Distance metric to equip the vector space with.
        angles : {'phi', 'psi', 'chi', 'omega', 'psi/psi', etc...}
            A slash separated list of strings specifying the types of angles to 
            compute per residue. The choices are 'phi', 'psi', 'chi', and 'omega',
            or any combination thereof
        p : int, optional
            p-norm order, used for metric='minkowski'
        V : ndarray, optional
            variances, used for metric='seuclidean'
        VI : ndarray, optional
            inverse covariance matrix, used for metric='mahalanobi'
        indices : ndarray, optional
            N x 4 numpy array of indices to be considered as dihedral angles. If
            provided, this overrrides the angles argument. The semantics of the
            array are that each row, indices[i], is an array of length 4 giving
            (in order) the indices of 4 atoms that together form a dihedral you
            want to monitor.
        
        See Also
        --------
        fast_cdist
        fast_pdist
        scipy.spatial.distance
        
        """
        super(Dihedral, self).__init__(metric, p, V, VI)
        self.angles = angles
        self.indices = indices
        
        if indices is not None:
            if not isinstance(indices, np.ndarray):
                raise ValueError('indices must be a numpy array')
            if not indices.ndim == 2:
                raise ValueError('indices must be 2D')
            if not indices.dtype == np.int:
                raise ValueError('indices must contain ints')
            if not indices.shape[1] == 4:
                raise ValueError('indices must be N x 4')
            logger.warning('OVERRIDING angles=%s and using custom indices instead', angles)
    
    def __repr__(self):
        "String representation of the object"
        return 'metrics.Dihedral(metric=%s, p=%s, angles=%s)' % (self.metric, self.p, self.angles)
    
    def prepare_trajectory(self, trajectory):
        """Prepare the dihedral angle representation of a trajectory, suitable
        for distance calculations.
        
        Parameters
        ----------
        trajectory : msmbuilder.Trajectory
            An MSMBuilder trajectory to prepare
        
        Returns
        -------
        projected_angles : ndarray
            A 2D array of dimension len(trajectory) x (2*number of dihedral
            angles per frame), such that in each row, the first half of the entries
            contain the cosine of the dihedral angles and the later dihedral angles
            contain the sine of the dihedral angles. This transform is necessary so
            that distance calculations preserve the periodic symmetry.
        """
        
        traj_length = len(trajectory['XYZList'])
        
        if self.indices is None:
            indices = _dihedralcalc.get_indices(trajectory, self.angles)
        else:
            indices = self.indices
        
        dihedrals = _dihedralcalc.compute_dihedrals(trajectory, indices, degrees=False)
        
        # these dihedrals go between -pi and pi but obviously because of the
        # periodicity, when we take distances we want the distance between -179
        # and +179 to be very close, so we need to do a little transform
        
        num_dihedrals = dihedrals.shape[1]
        transformed = np.empty((traj_length, 2 * num_dihedrals))
        transformed[:, 0:num_dihedrals] = np.cos(dihedrals)
        transformed[:, num_dihedrals:2*num_dihedrals] = np.sin(dihedrals)
        
        return np.double(transformed)
