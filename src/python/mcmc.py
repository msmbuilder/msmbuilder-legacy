import abc
import pymc
import numpy as np
import scipy.linalg
from msmbuilder import MSMLib

# Default values of prior parameters for Dirichlet and Exponential RVs
ALPHA = 1.0 # 1 pseudocount
BETA = 1.0E-6  # A nearly uninformative exponential.

def log_likelihood_T(C, T):
    """Get the log likelihood of a transition matrix given counts.

    Parameters
    ----------
    C : ndarray
        Count matrix
    T : ndarray
        Transition matrix

    Returns
    -------
    log_likelihood : float
        log-likelihood
    """    
    return np.sum(C * np.log(T))

class Sampler():
    """Base class for all MCMC samplers, provides `sample()` function."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, C):
        """Create Sampler and store the counts and num_states."""
        self.C = C
        self.num_states = C.shape[0]

    def sample(self, num_steps, thin=1, burn=0, filename=None):
        """Sample the MCMC chain and store the results.

        Parameters
        ----------
        num_steps : int
            How many MCMC samples to generate
        thin : int, optional, default 1
            Subsample MCMC chain by this amount.
        burn : int, optional, default 0
            Discard first `burn` MCMC samples.  
        filename : str, optional
            If not None, store results as HDF file rather than in RAM.

        """        
        if filename == None:
            db = "ram"
        else:
            db = "hdf5"
            
        self.mcmc = pymc.MCMC(self, db=db, dbname=filename)
        self.mcmc.sample(num_steps, thin=thin, burn=burn)    


class TransitionSampler(Sampler):
    """Sample transition matrices with Dirichlet prior."""
    def __init__(self, C, alpha=ALPHA):
        """Create an object to sample transition matrices.

        Parameters
        ----------
        C : ndarray
            Count matrix
        alpha : float
            Strength of Dirichlet prior.
        """
        Sampler.__init__(self, C)

        initial = MSMLib.estimate_transition_matrix(C)
        self.T_incomplete = np.array([pymc.Dirichlet("T%d_incomplete" % i, alpha * np.ones(self.num_states), value=initial[i,:-1]) for i in xrange(self.num_states)])
        self.T0 = np.array([pymc.CompletedDirichlet("T%d" % i,self.T_incomplete[i])[0] for i in xrange(self.num_states)])

        @pymc.dtrm
        def T(T0=self.T0):
            T = np.zeros((self.num_states, self.num_states))
            for i in xrange(self.num_states):
                T[i] = T0[i]
            return T
        self.T = T

        @pymc.potential
        def log_likelihood(T=self.T):
            return log_likelihood_T(self.C, T)
        self.log_likelihood = log_likelihood

        
class RateSampler(Sampler):
    """Sample rate matrices with an exponential prior."""
    def __init__(self, C, beta=BETA):
        """Create an object to sample rate matrices.

        Parameters
        ----------
        C : ndarray
            Count matrix
        beta : float
            Parameter of exponential prior. 
        """
        Sampler.__init__(self, C)

        T = MSMLib.estimate_transition_matrix(C)
        K0 = scipy.linalg.logm(T)
        np.fill_diagonal(K0, np.zeros(self.num_states))
        self.K_unnormalized = pymc.Exponential("K_unnormalized", beta, size=(self.num_states, self.num_states), value=K0)
 
        @pymc.dtrm
        def K(K_unnormalized=self.K_unnormalized):
            K = K_unnormalized.copy()
            diags = K.diagonal() - K.sum(1)
            np.fill_diagonal(K, diags)
            return K
        self.K = K

        @pymc.dtrm
        def T(K=self.K):
            T = scipy.linalg.expm(K)
            return T
        self.T = T

        @pymc.potential
        def log_likelihood(T=self.T):
            return log_likelihood_T(self.C, T)
        self.log_likelihood = log_likelihood
        

class ReversibleTransitionSampler(Sampler):
    """Sample reversible transition matrices."""
    def __init__(self, C):
        """Create an object to sample reversible transition matrices.

        Parameters
        ----------
        C : ndarray
            Count matrix
            
        Notes
        -----
        To parameterize reversible transition matrices, we work with
        symmetric count matrices X.  The parameters are the upper (inclusive)
        triangle of X, which we (arbitrarily) throw a uniform(0,1) prior on.        
        """
        Sampler.__init__(self, C)
        
        C_sym = C + C.T
        C_sym /= C_sym.sum()
        
        self.num_states = C.shape[0]
        self.num_tril = np.tril_indices_from(C)[0].shape[0]
        
        self.tril_indices = np.tril_indices_from(C)
        self.X_flat = pymc.Uniform("X_flat", 0, 1, size=(self.num_tril),value=C_sym[self.tril_indices])       
        
        @pymc.dtrm
        def X(X_flat=self.X_flat):

            X = np.zeros((self.num_states, self.num_states))

            X[self.tril_indices] = X_flat
            X.T[self.tril_indices] = X_flat            
            
            return X
        self.X = X

        @pymc.dtrm
        def T(X=self.X):
            T = MSMLib.estimate_transition_matrix(X)
            return T
        self.T = T

        @pymc.potential
        def log_likelihood(T=self.T):
            return log_likelihood_T(self.C, T)
        self.log_likelihood = log_likelihood


class ReversibleRateSampler(Sampler):
    def __init__(self, C, beta=BETA, alpha=ALPHA):
        """Create an object to sample reversible rate matrices.

        Parameters
        ----------
        C : ndarray
            Count matrix
        beta : float
            Parameter of exponential prior
        alpha : float
            Parameter of Dirichlet prior

        Notes
        -----
        To parameterize reversible rate matrices, we work with both the
        equilibrium populations (p) and the symmetrized rate matrix, 
        S = (diag(p**-0.5)) K (diag(p**0.5))
        
        We model the upper triangle (exclusive) of S using independent 
        exponential variables.  We model the equilibrium populations 
        as a Dirichlet variable.  
        """ 
        Sampler.__init__(self, C)

        self.num_tril = np.tril_indices_from(C,-1)[0].shape[0]

        T = MSMLib.estimate_transition_matrix(C + C.T)
        K0 = scipy.linalg.logm(T)
        l,v = np.linalg.eig(K0.T)
        p0 = v[:,l.argmax()]
        p0 /= p0.sum()        
        
        self.tril_indices = np.tril_indices_from(C, -1)
        self.S_flat = pymc.Exponential("S_tril", beta, size=(self.num_tril), value=K0[self.tril_indices])        
        
        self.p_incomplete = pymc.Dirichlet("p_incomplete", alpha * np.ones(self.num_states), value=p0[:-1])
        self.p = pymc.CompletedDirichlet("p",self.p_incomplete)[0]

        @pymc.dtrm
        def S(S_flat=self.S_flat):

            S = np.zeros((self.num_states, self.num_states))

            S[self.tril_indices] = S_flat
            S.T[self.tril_indices] = S_flat            

            return S
        self.S = S

        @pymc.dtrm
        def K(S=self.S, p=self.p):
            D_pre = np.diag(p ** 0.5)
            D_post = np.diag(p ** -0.5)
            K = (D_pre.dot(S)).dot(D_post)
            
            np.fill_diagonal(K, -1 * K.sum(1))
            
            return K
        self.K = K

        @pymc.dtrm
        def T(K=self.K):
            T = scipy.linalg.expm(K)
            return T
        self.T = T

        @pymc.potential
        def log_likelihood(T=self.T):
            return log_likelihood_T(self.C, T)
        self.log_likelihood = log_likelihood
        
