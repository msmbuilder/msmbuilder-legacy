import pymc  # pymc 2.2
import numpy as np
import scipy.linalg

class TransitionSampler():
    def __init__(self, C):
        self.C = C
        self.num_states = C.shape[0]
        initial = (1 / float(self.num_states)) * np.ones((self.num_states, self.num_states - 1))
        self.T_partial = pymc.Uninformative("T_partial",size=(self.num_states, self.num_states - 1),value=initial)

        @pymc.dtrm
        def T(T_partial=self.T_partial):
            T = np.zeros((self.num_states, self.num_states))
            T[:,:-1] = T_partial
            T[:,-1] = 1 - T_partial.sum(1)
            return T
        self.T = T

        @pymc.potential
        def log_likelihood(T=self.T):
            return np.sum(self.C * np.log(T))
        self.log_likelihood = log_likelihood
        
        @pymc.potential
        def normalization_constraint(T=self.T):
            if T.min() < 0 or T.sum(1).max() > 1:
                return -1 * np.inf
            else:
                return 0.0
        self.normalization_constraint = normalization_constraint

    def sample(self, num_steps, thin=1, filename=None):

        if filename == None:
            db = "ram"
        else:
            db = "hdf5"
            
        self.mcmc = pymc.MCMC(self, db=db, dbname=filename)
        self.mcmc.sample(num_steps)
        
    def iterate_T(self):
        """Iterate over the sampled transition matrices."""
        Ti = self.mcmc.trace("T")
        for i, T in enumerate(Ti):
            yield T

class RateSampler():
    def __init__(self, C):
        self.C = C
        self.num_states = C.shape[0]
        initial = (1 / float(self.num_states)) * np.ones((self.num_states, self.num_states))
        self.K_unnormalized = pymc.Uninformative("K_unnormalized",size=(self.num_states, self.num_states),value=initial)

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
            return np.sum(self.C * np.log(T))
        self.log_likelihood = log_likelihood
        
        @pymc.potential
        def normalization_constraint(K=self.K):
            K = K.copy()
            np.fill_diagonal(K, np.zeros(self.num_states))
            if K.min() < 0.0:
                return -1 * np.inf
            else:
                return 0.0
        self.normalization_constraint = normalization_constraint

    def sample(self, num_steps, thin=1, filename=None):

        if filename == None:
            db = "ram"
        else:
            db = "hdf5"
            
        self.mcmc = pymc.MCMC(self, db=db, dbname=filename)
        self.mcmc.sample(num_steps)
        
    def iterate_K(self):
        """Iterate over the sampled transition matrices."""
        Ki = self.mcmc.trace("K")
        for i, K in enumerate(Ki):
            yield K

class ReversibleTransitionSampler():
    def __init__(self, C):
        self.C = C
        self.num_states = C.shape[0]
        initial = (1 / float(self.num_states)) * np.ones((self.num_states, self.num_states - 1))
        self.X_partial = pymc.Uninformative("X_partial",size=(self.num_states, self.num_states - 1),value= initial)

        @pymc.dtrm
        def X(X_partial=self.X_partial):
            X = np.zeros((self.num_states, self.num_states))
            X[:,:-1] = X_partial
            X[:,-1] = 1 - X_partial.sum(1)
            X = 0.5 * (X + X.T)
            return X
        self.X = X

        @pymc.dtrm
        def T(X=self.X):
            Ni = X.sum(1)
            D = np.diag(Ni**-1.)
            T = D.dot(X)
            return T
        self.T = T

        @pymc.potential
        def log_likelihood(T=self.T):
            return np.sum(self.C * np.log(T))
        self.log_likelihood = log_likelihood
        
        @pymc.potential
        def normalization_constraint(T=self.T):
            if T.min() < 0 or T.sum(1).max() > 1:
                return -1 * np.inf
            else:
                return 0.0
        self.normalization_constraint = normalization_constraint

    def sample(self, num_steps, thin=1):
        self.mcmc = pymc.MCMC(self)
        self.mcmc.sample(num_steps)
        return self.mcmc.trace("T")[:]
