from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from msmbuilder import msm_analysis, MSMLib
import numpy as np
import statsmodels
import statsmodels.tsa.stattools
from unittest import skipIf


@skipIf(int(statsmodels.__version__.split('.')[1]) <= 4, "Need developer version of statsmodels to use statsmodels.tsa.stattools.acf")
class test_msm_acf():
    def __init__(self):
        self.epsilon = 1E-7
        self.alpha = 0.001  # Confidence for uncertainty estimate
        # Testing is stochastic; we expect errors 0.1 % of the time.
        self.max_lag = 100
        self.times = np.arange(self.max_lag)
        self.num_steps = 100000

        self.C = np.array([[500, 2], [2, 50]])
        self.T = MSMLib.estimate_transition_matrix(self.C)
        self.state_traj = np.array(msm_analysis.sample(self.T, 0, self.num_steps))

    def compare_observable_to_statsmodels(self, observable_by_state):
        acf_msm = msm_analysis.msm_acf(self.T, observable_by_state, self.times)
        observable_traj = observable_by_state[self.state_traj]
        acf, errs = statsmodels.tsa.stattools.acf(observable_traj, nlags=self.max_lag - 1, fft=True, alpha=self.alpha)

        min_acf, max_acf = errs.T
        np.testing.assert_((acf_msm <= max_acf + self.epsilon).all())
        np.testing.assert_((min_acf <= acf_msm + self.epsilon).all())

    def test_1(self):
        observable_by_state = np.eye(2)[0]
        self.compare_observable_to_statsmodels(observable_by_state)

    def test_2(self):
        observable_by_state = np.array([0.25, 0.75])
        self.compare_observable_to_statsmodels(observable_by_state)
