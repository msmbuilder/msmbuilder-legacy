from msmbuilder import msm_analysis, MSMLib
import numpy as np
import statsmodels.tsa.stattools

class test_msm_acf():
    def __init__(self):
        self.epsilon = 1E-7
        self.alpha = 0.005
        self.max_lag = 10
        self.times = np.arange(self.max_lag)
        self.num_steps = 1000000        
        
        self.C = np.array([[100,2],[2,10]])
        self.T = MSMLib.estimate_transition_matrix(self.C)
        self.state_traj = np.array(msm_analysis.sample(self.T,0,self.num_steps))
    
    def test_1(self):
        obs = np.eye(2)[0]
        acf_msm = msm_analysis.msm_acf(self.T,obs,self.times)
        obs_traj = obs[self.state_traj]
        acf,errs = statsmodels.tsa.stattools.acf(obs_traj,nlags=self.max_lag - 1,fft=True,alpha=self.alpha)
        
        min_acf, max_acf = errs.T
        np.testing.assert_((acf_msm <= max_acf + self.epsilon).all())
        np.testing.assert_((min_acf <= acf_msm + self.epsilon).all())
    
    def test_2(self):
        obs = np.array([0.25,0.75])
        acf_msm = msm_analysis.msm_acf(self.T,obs,self.times)
        obs_traj = obs[self.state_traj]
        acf,errs = statsmodels.tsa.stattools.acf(obs_traj,nlags=self.max_lag - 1,fft=True,alpha=self.alpha)
        
        min_acf, max_acf = errs.T
        np.testing.assert_((acf_msm <= max_acf + self.epsilon).all())
        np.testing.assert_((min_acf <= acf_msm + self.epsilon).all())