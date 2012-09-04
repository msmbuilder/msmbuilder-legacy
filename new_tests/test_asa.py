import os, sys
import numpy as np
import numpy.testing as npt
from nose.tools import ok_, eq_

from common import fixtures_dir

from msmbuilder import Trajectory

from msmbuilder.geometry.asa import calculate_asa, ATOMIC_RADII


def test_asa_0():    
    # make one atom at the origin
    traj = {'XYZList': np.zeros((1,1,3), dtype=np.float32),
            'AtomNames': ['H']}
            
    probe_radius = 0.14
    calc_area = np.sum(calculate_asa(traj, probe_radius=probe_radius))
    true_area = 4 * np.pi * (ATOMIC_RADII['H'] + probe_radius)**2

    npt.assert_approx_equal(calc_area, true_area)

def test_asa_1():
    # two atoms
    traj = {'XYZList': np.zeros((1,2,3), dtype=np.float32),
            'AtomNames': ['H', 'H']}

    probe_radius = 0.14
    true  = 4 * np.pi * (ATOMIC_RADII['H'] + probe_radius)**2

    # when atoms are closer than 2e-5, there seems to be a bug.
    # note that you should never actually have a case where atoms are this close
    # but nonetheless I'm adding a check for this in the implementation -- to make
    # it crash if the atoms are too close, as opposed to giving you wrong results
    separations = np.linspace(2.0e-5, probe_radius*2 + ATOMIC_RADII['H']*2, 10)
    areas = np.zeros_like(separations)

    # check the asa as we vary the separation
    for i, sep in enumerate(separations):
        traj['XYZList'][0, 0, 1] = sep
        areas[i] = np.sum(calculate_asa(traj, probe_radius=probe_radius))
    
    npt.assert_approx_equal(areas[0], true)
    npt.assert_approx_equal(areas[-1], 2*true)
    # make sure that areas is increasing
    npt.assert_array_less(areas[0:8], areas[1:9])


def test_asa_2():
    t = Trajectory.LoadTrajectoryFile(os.path.join(fixtures_dir(), 'trj0.lh5'))
    val1 = np.sum(calculate_asa(t[0])) # calculate only frame 0
    val2 = np.sum(calculate_asa(t)[0]) # calculate on all frames
    true_frame_0_asa = 3.55564826906
    
    npt.assert_approx_equal(true_frame_0_asa, val1)
    npt.assert_approx_equal(true_frame_0_asa, val2)
    
