from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import nose
import functools
import numpy as np
from nose.tools import eq_, ok_, raises
from numpy.testing import dec
import mdtraj as md

import os
import inspect

def load_traj():
    "Load up a 3 frame, 4 atom trajectory for testing"
    
    empty = np.array([])
    frame0 = np.array([[[1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3],
                    [4, 4, 4]]])
    frame1 = np.array([[[-1, -1, -1],
                        [-2, -2, -2],
                        [-3, -3, -3],
                        [-4, -4, -4]]])
    frame2 = np.array([[[1, 2, 1],
                        [3, 2, 2],
                        [3, 4, 3],
                        [5, 4, 4]]])
    return md.Trajectory(xyz=np.concatenate((frame0, frame1, frame2)), topology=None)
