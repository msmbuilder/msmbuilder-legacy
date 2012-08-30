import os
import numpy as np
from msmbuilder.geometry import rg as rgcalc
from msmbuilder.Project import Project

import numpy.testing as npt
from common import fixtures_dir

def reference_rg(xyzlist):
    '''Return the radius of gyration of every frame in a xyzlist'''
    
    traj_length = len(xyzlist)
    Rg = np.zeros(traj_length)
    for i in xrange(traj_length):
        XYZ = xyzlist[i, :, :]
        mu = XYZ.mean(0)
        XYZ2 = XYZ - np.tile(mu, (len(XYZ), 1))
        Rg[i] = (XYZ2**2.0).mean()**0.5
    
    return Rg

def test_rg_1():
    project = Project.LoadFromHDF(os.path.join(fixtures_dir(),
        'ProjectInfo.h5'))
    traj = project.LoadTraj(0)
    xyzlist = traj['XYZList']

    a = rgcalc.calculate_rg(xyzlist)
    b = reference_rg(xyzlist)
    
    npt.assert_array_almost_equal(a, b)
