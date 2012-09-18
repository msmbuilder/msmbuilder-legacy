import numpy as np
import numpy.testing as npt
import IPython as ip
# LPRMSD is a class defined in lprmsd.py which is the lprmsd package.
from lprmsd import LPRMSD
from msmbuilder import Trajectory
from msmbuilder.metrics import RMSD

def test_lprmsd():
    t = Trajectory.load_trajectory_file('trj0.lh5')

    MyIdx = np.array([1, 4, 5, 6, 8, 10, 14, 15, 16, 18])

    lprmsd = LPRMSD(atomindices=MyIdx, debug=True)

    lptraj = lprmsd.prepare_trajectory(t)

    dists = lprmsd.one_to_all(lptraj, lptraj, 0)

    lprmsd_alt = LPRMSD(atomindices=MyIdx, altindices=MyIdx, debug=True)
    lptraj_alt = lprmsd_alt.prepare_trajectory(t)
    dists_alt = lprmsd_alt.one_to_all(lptraj_alt, lptraj_alt, 0)

    rmsd = RMSD(atomindices=MyIdx)
    reftraj = rmsd.prepare_trajectory(t)
    ref_dists = rmsd.one_to_all(reftraj, reftraj, 0)

    
    npt.assert_array_almost_equal(dists, ref_dists)
    npt.assert_array_almost_equal(dists_alt, ref_dists)
