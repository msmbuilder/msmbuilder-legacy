
import numpy as np
import re, sys, os
from time import time
import logging
from msmbuilder import io

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class tICA(object):
    """
    tICA is a class for calculating the matrices required to do time-structure
    based independent component analysis (tICA). It can be
    used to calculate both the time-lag correlation matrix and covariance
    matrix. The advantage it has is that you can calculate the matrix for a 
    large dataset by "training" smaller pieces of the dataset at a time. 

    Notes
    -----

    It can be shown that the time-lag correlation matrix is the same as:

    C = E[Outer(X[t], X[t+lag])] - Outer(E[X[t]], E[X[t+lag]])

    Because of this it is possible to calculate running sums corresponding 
    to variables A, B, D:

    A = E[X[t]]
    B = E[X[t+lag]]
    D = E[Outer(X[t], X[t+lag])]

    Then at the end we can calculate C:

    C = D - Outer(A, B)

    Finally we can get a symmetrized C' from our estimate of C, for
    example by adding the transpose:

    C' = (C + C^T) / 2
     
    There is, in fact, an MLE estimator for ech matrix C, and S:

    S = E[Outer(X[t], X[t])]

    The MLE estimators are:

    \mu = 1 / (2(N - lag)) \sum_{t=1}^{N - lag} X[t] + X[t + lag]

    C = 1 / (2(N - lag)) * \sum_{t=1}^{N - lag} Outer(X[t] - \mu, X[t + lag] - \mu) + Outer(X[t + lag] - \mu, X[t] - \mu)

    S = 1 / (2(N - lag)) * \sum_{t=1}^{N - lag} Outer(X[t] - \mu, X[t] - \mu) + Outer(X[t + lag] - \mu, X[t + lag] - \mu)

    """

    def __init__(self, lag, calc_cov_mat=True, size=None):
        """
        Create an empty tICA object.

        To add data to the object, use the train method.

        Parameters
        ----------
        lag: int
            The lag to use in calculating the time-lag correlation
            matrix. If zero, then only the covariance matrix is
            calculated
        calc_cov_mat: bool, optional
            if lag > 0, then will also calculate the covariance matrix
        size: int, optional
            the size is the number of coordinates for the vector
            representation of the protein. If None, then the first
            trained vector will be used to initialize it.
        """
        
        self.corrs = None
        self.sum_t = None
        self.sum_t_dt = None
        # The above containers hold a running sum that is used to 
        # calculate the time-lag correlation matrix as well as the
        # covariance matrix

        self.corrs_lag0 = None  # needed for calculating the covariance
                                # matrix       
        self.sum_all = None

        self.trained_frames = 0
        self.total_frames = 0
        # Track how many frames we've trained

        self.lag=int(lag)
        if self.lag < 0:
            raise Exception("lag must be non-negative.")
        elif self.lag == 0:  # If we have lag=0 then we don't need to
                             # calculate the covariance matrix twice
            self.calc_cov_mat = False
        else:
            self.calc_cov_mat = calc_cov_mat

        self.size = size
        if not self.size is None:
            self.set_size(size)

        # containers for the solutions:
        self.timelag_corr_mat = None
        self.cov_mat = None
        self.vals = None
        self.vecs = None
    
        self._sorted = False
            

    def set_size(self, N):
        """
        Set the size of the matrix.

        Parameters
        ----------
        N : int
            The size of the square matrix will be (N, N)

        """

        self.size = N

        self.corrs = np.zeros((N,N), dtype=float)
        self.sum_t = np.zeros(N, dtype=float)
        self.sum_t_dt = np.zeros(N, dtype=float)
        self.sum_all = np.zeros(N, dtype=float)

        if self.calc_cov_mat:
            self.corrs_lag0_t = np.zeros((N, N), dtype=float)
            self.corrs_lag0_t_dt = np.zeros((N, N), dtype=float)


    def train(self, data_vector):
        a=time()  # For debugging we are tracking the time each step takes

        if self.size is None:  
        # then we haven't started yet, so set up the containers
            self.set_size(data_vector.shape[1])

        if data_vector.shape[1] != self.size:
            raise Exception("Input vector is not the right size. axis=1 should "
                            "be length %d. Vector has shape %s" %
                            (self.size, str(data_vector.shape)))

        if data_vector.shape[0] <= self.lag:
            logger.warn("Data vector is too short (%d) "
                        "for this lag (%d)", data_vector.shape[0],self.lag)
            return

        b=time()

        if self.lag != 0:
            self.corrs += data_vector[:-self.lag].T.dot(data_vector[self.lag:])
            self.sum_t += data_vector[:-self.lag].sum(axis=0)
            self.sum_t_dt += data_vector[self.lag:].sum(axis=0)
        else:
            self.corrs += data_vector.T.dot(data_vector)
            self.sum_t += data_vector.sum(axis=0)
            self.sum_t_dt += self.sum_t

        if self.calc_cov_mat:
            self.corrs_lag0_t += data_vector[:-self.lag].T.dot(data_vector[:-self.lag])
            self.corrs_lag0_t_dt += data_vector[self.lag:].T.dot(data_vector[self.lag:])
            self.sum_all += data_vector.sum(axis=0)
            self.total_frames += data_vector.shape[0]

        self.trained_frames += data_vector.shape[0] - self.lag  
        # this accounts for us having finite trajectories, so we really are 
        #  only calculating expectation values over N - \Delta t total samples

        c=time()

        logger.debug("Setup: %f, Corrs: %f" %(b-a, c-b))
        # Probably should just get rid of this..


    def get_current_estimate(self):
        """Calculate the current estimate of the time-lag correlation
        matrix and the covariance matrix (if asked for).

        Currently, this is done by symmetrizing the sample time-lag
        correlation matrix, which can cause problems!
        
        """
        two_N = 2. * float(self.trained_frames)
        # ^^ denominator in all of these expressions...
        mle_mean = (self.sum_t + self.sum_t_dt) / two_N
        outer_means = np.outer(mle_mean, mle_mean)
        
        time_lag_corr = (self.corrs + self.corrs.T) / two_N

        timelag_corr_mat = time_lag_corr - outer_means

        self.timelag_corr_mat = timelag_corr_mat

        if self.calc_cov_mat:

            cov_mat = (self.corrs_lag0_t + self.corrs_lag0_t_dt) / two_N
            cov_mat -= np.outer(mle_mean, mle_mean)

            return timelag_corr_mat, cov_mat

            self.cov_mat = cov_mat

        return timelag_corr_mat

    
    def _sort(self):
        """
        sort the eigenvectors by their eigenvalues.
        """
        if self.vals is None:
            self.solve()

        ind = np.argsort(self.vals)[::-1] 
        # in order of decreasing value
        self.vals = self.vals[ind]
        self.vecs = self.vecs[:, ind]

        self._sorted = True


    def solve(self, pca_cutoff=0):
        """
        Solve the eigenvalue problem. We can translate into the
        PCA space and remove directions that have zero variance.
        
        If there are directions with zero variance, then the tICA
        eigenvalues will be complex or greater than one.

        Parameters:
        -----------
        pca_cutoff : float, optional
            pca_cutoff to throw out PCs with variance less than this
            cutoff. Default is zero, but you should really check
            your covariance matrix to see if you need this.

        """
        
        if self.timelag_corr_mat is None or self.cov_mat is None:
            self.get_current_estimate()

        # should really add check if we're just doing PCA, but I
        # don't know why anyone would use this class to do PCA...
        # maybe I should just remove that ability...

        if pca_cutoff <= 0:
            lhs = self.timelag_corr_mat
            rhs = self.cov_mat

        else:
            pca_vals, pca_vecs = np.linalg.eigh(self.cov_mat)

            good_ind = np.where(pca_vals > pca_cutoff)[0]

            pca_vals = pca_vals[good_ind]
            pca_vecs = pca_vecs[:, good_ind]

            lhs = pca_vecs.T.dot(self.timelag_corr_mat).dot(pca_vecs)
            rhs = pca_vecs.T.dot(self.cov_mat).dot(pca_vecs)

        vals, vecs = scipy.linalg.eig(lhs, b=rhs)

        if pca_cutoff <= 0:
            self.vals = vals
            self.vecs = pca_vecs.dot(vecs)

        else:
            self.vals = vals
            self.vecs = vecs

        if np.abs(self.vals.imag).max() > 1E-10:
            logger.warn("you have non-real eigenvalues. This usually means "
                        "you need to throw out some coordinates by doing tICA "
                        "in PCA space.")

        else:
            self.vals = self.vals.real

        if np.abs(self.vecs.imag).max() > 1E-10:
            logger.warn("you have non-real eigenvector entries...")

        else:
            self.vecs = self.vecs.real
        
        self._sort()


    def project(self, trajectory=None, prep_trajectory=None, which=None):
        """
        project a trajectory (or prepared trajectory) onto a subset of
        the tICA eigenvectors.

        Parameters:
        -----------
        trajectory : msmbuilder.Trajectory, optional
            trajectory object (can also pass a prepared trajectory instead)
        prep_trajectory : np.ndarray, optional
            prepared trajectory
        which : np.ndarray
            which eigenvectors to project onto

        Returns:
        --------
        proj_trajectory : np.ndarray
            projected trajectory (n_points, n_tICs)
        """
        if not self._sorted():
            self._sort()

        if prep_trajectory is None:
            if trajectory is None:
                raise Exception("must pass one of trajectory or prep_trajectory")
            prep_trajectory = self.metric.prepare_trajectory(trajectory)

        if which is None:
            raise Exception("must pass 'which' to indicate which tICs to project onto")
        
        which = np.array(which).flatten().astype(int)

        proj_trajectory = prep_trajectory.dot(self.vecs[:, which])
    
        return proj_trajectory


def load(tica_fn, metric):
    """
    load a tICA solution to use in projecting data.

    Parameters:
    -----------
    tica_fn : str
        filename pointing to tICA solutions
    metric : metrics.Vectorized subclass instance
        metric used to prepare trajectories

    """
    
    # the only variables we need to save are the two matrices
    # and the eigenvectors / values
    
    f = io.loadh(tica_fn)

    tica_obj = tICA(f['dt'])

    tica_obj.timelag_corr_mat = f['timelag_corr_mat']
    tica_obj.cov_mat = f['cov_mat']

    tica_obj.vals = f['vals']
    tica_obj.vecs = f['vecs']

    tica_obj._sort()

    return tica_obj
    
