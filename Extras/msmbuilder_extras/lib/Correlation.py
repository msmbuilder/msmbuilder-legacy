# This file is part of MSMBuilder.
#
# Copyright 2011 Stanford University
#
# MSMBuilder is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

import numpy as np
from msmbuilder import msm_analysis
from msmbuilder.utils import fft_acf    

def raw_msm_correlation(trans_matrix, observable, assignments, n_steps=10000, starting_state=0):
    """Calculate an autocorrelation function from an MSM.

    Parameters
    ----------
    trans_matrix : sparse or dense matrix
        Transition matrix
    observable : np.ndarray, shape=[n_trajs, max_n_frames]
        Value of the observable in each conf
    assignments : np.ndarray, shape=[n_trajs, max_n_frames]
        State membership for each conf
    n_steps : int
        Number of steps to simulate
    starting_state : int
        State to start the trajectory from

    
    Notes
    -----
    This function works by first generating a 'sample' trajectory from the MSM. This
    approach is necessary as it allows treatment of intra-state dynamics.
    
    Returns
    -------
    correlation : np.ndarray, shape=[n_steps]
        The autocorrelation of the observable
    traj : np.ndarray, shape=[n_steps]
        The simulated trajectory, represented as a sequence of states
    obs_traj : np.ndarray, shape=[n_steps]
        The observable signal, as a sequence of values from traj
    """

    traj = msm_analysis.sample(trans_matrix, starting_state, n_steps)

    # NOTE: I'm not sure if this MaxInt is right. (RTM 9/6)
    MaxInt = np.ones(assignments.shape).sum()

    obs_traj = np.array( [observable[ np.where( assignments  == State ) ].take( [ np.random.randint( MaxInt ) ], mode='wrap' ) for State in traj ] )

    corr = fft_acf(obs_traj)

    return corr, traj, obs_traj

