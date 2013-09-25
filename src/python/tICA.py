
import numpy as np
import re, sys, os
from time import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class tICA(object):
    """
    tICA is a class for calculating covariance matrices. It can be
    used to calculate both the time-lag correlation matrix and covariance
    matrix. The advantage it has is that you can calculate the matrix for a 
    large dataset by "training" pieces of the dataset at a time. 

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

        current_estimate = time_lag_corr - outer_means

        if self.calc_cov_mat:

            cov_mat = (self.corrs_lag0_t + self.corrs_lag0_t_dt) / two_N
            cov_mat -= np.outer(mle_mean, mle_mean)

            return current_estimate, cov_mat

        return current_estimate

