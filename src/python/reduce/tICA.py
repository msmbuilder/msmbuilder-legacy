import numpy as np
import scipy.linalg
from time import time
import cPickle
import logging
from mdtraj import io
from msmbuilder.metrics import Vectorized
from msmbuilder.reduce import AbstractDimReduction

logger = logging.getLogger(__name__)

class tICA(AbstractDimReduction):
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

    def __init__(self, lag, calc_cov_mat=True, prep_metric=None, size=None):
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
        prep_metric: msmbuilder.metrics.Vectorized subclass instance, optional
            metric to use to prepare trajectories. If not specified, then
            you must pass prepared trajectories to the train method, via
            the kwarg "prep_trajectory"
        size: int, optional
            the size is the number of coordinates for the vector
            representation of the protein. If None, then the first
            trained vector will be used to initialize it.
            
        Notes
        -----
        
        To load an already constructed tICA object, use `tICA.load()`.
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

        if prep_metric is None:
            self.prep_metric = None
            logger.warn("no metric specified, you must pass prepared"
                " trajectories to the train and project methods")

        else:
            if not isinstance(prep_metric, Vectorized):
                raise Exception("prep_metric must be an instance of a "
                    "subclass of msmbuilder.metrics.Vectorized")

            self.prep_metric = prep_metric

        self.size = size
        if not self.size is None:
            self.initialze(size)

        # containers for the solutions:
        self.timelag_corr_mat = None
        self.cov_mat = None
        self.vals = None
        self.vecs = None
    
        self._sorted = False
            

    def initialize(self, size):
        """
        initialize the containers for the calculation

        Parameters
        ----------
        size : int
            The size of the square matrix will be (size, size)
        """

        self.size = size

        self.corrs = np.zeros((size, size), dtype=float)
        self.sum_t = np.zeros(size, dtype=float)
        self.sum_t_dt = np.zeros(size, dtype=float)
        self.sum_all = np.zeros(size, dtype=float)

        if self.calc_cov_mat:
            self.corrs_lag0_t = np.zeros((size, size), dtype=float)
            self.corrs_lag0_t_dt = np.zeros((size, size), dtype=float)


    def train(self, trajectory=None, prep_trajectory=None):
        """
        add a trajectory to the calculation

        Parameters:
        -----------
        trajectory: msmbuilder.Trajectory, optional
            trajectory object
        prep_trajectory: np.ndarray, optional
            prepared trajectory object

        Remarks:
        --------
        must input one of trajectory or prep_trajectory (if both
        are given, then prep_trajectory is used.)
        """

        if not prep_trajectory is None:
            data_vector = prep_trajectory

        elif not trajectory is None:
            data_vector = self.prep_metric.prepare_trajectory(trajectory)

        else:
            raise Exception("need to input one of trajectory or prep_trajectory")

        a=time()  # For debugging we are tracking the time each step takes

        if self.size is None:  
        # then we haven't started yet, so set up the containers
            self.initialize(size=data_vector.shape[1])

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

        These estimates come from an MLE argument assuming that the data {X_t, X_t+dt}
        are distributed as a multivariate normal. Of course, this assumption 
        is not very true, but this is merely one way to enforce that the 
        timelag correlation matrix is symmetric. 

        The MLE has nice properties, as well, such as the eigenvalues that result
        from solving the tICA equation are always bounded between -1 and 1, which
        is not the case when one merely symmetrizes the timelag correlation matrix
        while estimating the covariance matrix and mean in the usual manner.

        See Shukla, D et. al. In Preparation for details, or email Christian
        Schwantes (schwancr@stanford.edu).
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

            self.cov_mat = cov_mat

            return timelag_corr_mat, cov_mat

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
            self.vecs = vecs

        else:
            self.vals = vals
            self.vecs = pca_vecs.dot(vecs)

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
        trajectory : mdtraj.Trajectory, optional
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
        if not self._sorted:
            self._sort()

        if prep_trajectory is None:
            if trajectory is None:
                raise Exception("must pass one of trajectory or prep_trajectory")
            prep_trajectory = self.prep_metric.prepare_trajectory(trajectory)

        if which is None:
            raise Exception("must pass 'which' to indicate which tICs to project onto")
        
        which = np.array(which).flatten().astype(int)

        proj_trajectory = prep_trajectory.dot(self.vecs[:, which])
    
        return proj_trajectory


    def save(self, output):
        """
        save the results to file
        
        Parameters:
        -----------
        output : str
            output filename (.h5)
        """
        
        metric_string = cPickle.dumps(self.prep_metric)  # Serialize metric used to calculate tICA input.
        
        io.saveh(output, timelag_corr_mat=self.timelag_corr_mat,
            cov_mat=self.cov_mat, lag=np.array([self.lag]), vals=self.vals,
            vecs=self.vecs, metric_string=np.array([metric_string]))

    @classmethod
    def load(cls, tica_fn):
        """
        load a tICA solution to use in projecting data.

        Parameters:
        -----------
        tica_fn : str
            filename pointing to tICA solutions

        """
        # the only variables we need to save are the two matrices
        # and the eigenvectors / values as well as the lag time
        
        logger.warn("NOTE: You can only use the tICA solution, you will "
                    "not be able to continue adding data")
        f = io.loadh(tica_fn)
        
        metric = cPickle.loads(f["metric_string"][0])

        tica_obj = cls(f['lag'][0], prep_metric=metric)
        # lag entry is an array... with a single item

        tica_obj.timelag_corr_mat = f['timelag_corr_mat']
        tica_obj.cov_mat = f['cov_mat']

        tica_obj.vals = f['vals']
        tica_obj.vecs = f['vecs']

        tica_obj._sort()

        return tica_obj
        
