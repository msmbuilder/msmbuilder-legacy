import pymc  # pymc 2.2
import numpy as np
import scipy.linalg
from msmbuilder import MSMLib

class TransitionSampler():
    def __init__(self, C, alpha=1.0):
        self.C = C
        self.num_states = C.shape[0]

        self.T_incomplete = np.array([pymc.Dirichlet("T%d_incomplete" % i, alpha * np.ones(self.num_states)) for i in xrange(self.num_states)])
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
            return np.sum(self.C * np.log(T))
        self.log_likelihood = log_likelihood

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
        initial_value = (1 / float(self.num_states)) * np.ones((self.num_states, self.num_states))
        self.K_unnormalized = pymc.Uninformative("K_unnormalized",size=(self.num_states, self.num_states),value=initial_value)

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
        self.N = C.sum()
        
        self.num_states = C.shape[0]
        self.num_tril = np.tril_indices_from(C)[0].shape[0]
        
        self.X_flat = pymc.Uninformative("S_tril",size=(self.num_tril),value=np.ones(self.num_tril))
        self.tril_indices = np.tril_indices_from(C)
        
        self.p_incomplete = pymc.Dirichlet("p_incomplete", np.ones(self.num_states))
        self.p = pymc.CompletedDirichlet("p",self.p_incomplete)[0]

        @pymc.dtrm
        def X(X_flat=self.X_flat):

            X = np.zeros((self.num_states, self.num_states))

            X[self.tril_indices] = X_flat
            X.T[self.tril_indices] = X_flat            

            X /= X.sum()
            X *= self.N
            
            return X
        self.X = X

        @pymc.dtrm
        def T(X=self.X):
            T = MSMLib.estimate_transition_matrix(X)
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


class ReversibleRateSampler():
    def __init__(self, C):
        self.C = C
        self.num_states = C.shape[0]
        self.num_tril = np.tril_indices_from(C,-1)[0].shape[0]
        
        self.S_flat = pymc.Uninformative("S_tril",size=(self.num_tril),value=np.ones(self.num_tril))
        self.tril_indices = np.tril_indices_from(C,-1)
        
        self.p_incomplete = pymc.Dirichlet("p_incomplete", np.ones(self.num_states))
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
