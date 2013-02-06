import nose
import functools
import numpy as np
from msmbuilder import Trajectory
from nose.tools import eq_, ok_, raises
from numpy.testing import dec

import os
import inspect

def load_traj():
    "Load up a 3 frame, 4 atom trajectory for testing"
    
    empty = np.array([])
    frame0 = np.array([[1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3],
                    [4, 4, 4]])
    frame1 = np.array([[-1, -1, -1],
                        [-2, -2, -2],
                        [-3, -3, -3],
                        [-4, -4, -4]])
    frame2 = np.array([[1, 2, 1],
                        [3, 2, 2],
                        [3, 4, 3],
                        [5, 4, 4]])
    
    traj = Trajectory({'ChainID': empty,
        'AtomNames': empty,
        'ResidueNames': empty,
        'ResidueID': empty,
        'AtomID': empty,
        'XYZList': np.empty(shape=(3,4,3))})
    traj['XYZList'][0] = frame0
    traj['XYZList'][1] = frame1
    traj['XYZList'][2] = frame2
    
    return traj