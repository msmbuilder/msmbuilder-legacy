import numpy as np
from msmbuilder.testing import eq
from msmbuilder import mcmc, MSMLib
import scipy.linalg
from nose.tools import nottest
from nose.plugins.skip import SkipTest

SKIP_TESTS = True

num_states = 2
num_steps = 1000000
thin = 500
burn = 50000

C = 1000000 * np.ones((num_states,num_states))
C[0,0] *= 1000
C[0,1] += 1

T0 = MSMLib.estimate_transition_matrix(C)
X0 = MSMLib.mle_reversible_count_matrix(scipy.sparse.csr_matrix(C)).toarray()
T0_rev = MSMLib.estimate_transition_matrix(X0)

def test_TransitionSampler():
    if SKIP_TESTS == True:
        raise(SkipTest("MCMC test skipped"))
    sampler = mcmc.TransitionSampler(C)
    sampler.sample(num_steps, thin=thin, burn=burn)
    T = sampler.mcmc.trace("T")[:].mean(0)
    eq(T0, T, decimal=5)
    
def test_ReversibleTransitionSampler():
    if SKIP_TESTS == True:
        raise(SkipTest("MCMC test skipped"))
    sampler = mcmc.ReversibleTransitionSampler(C)
    sampler.sample(1000000,thin=500, burn=burn)
    T_rev = sampler.mcmc.trace("T")[:].mean(0)
    eq(T0_rev, T_rev, decimal=5)
    
def test_RateSampler():
    if SKIP_TESTS == True:
        raise(SkipTest("MCMC test skipped"))
    sampler = mcmc.RateSampler(C)
    sampler.sample(1000000,thin=500, burn=burn)
    T_K = sampler.mcmc.trace("T")[:].mean(0)
    eq(T0, T_K, decimal=5)
    
def test_ReversibleRateSampler():
    if SKIP_TESTS == True:
        raise(SkipTest("MCMC test skipped"))
    sampler = mcmc.ReversibleRateSampler(C)
    sampler.sample(1000000,thin=500, burn=burn)
    T_K_rev = sampler.mcmc.trace("T")[:].mean(0)
    eq(T0_rev, T_K_rev, decimal=5)
