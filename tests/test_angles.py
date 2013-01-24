import itertools
import numpy as np
from msmbuilder.geometry.angle import bond_angles, __bond_angles
from msmbuilder.testing import *

def test_asa_0():
    conf = get('native.pdb')
    n_atoms = conf['XYZList'].shape[1]
    angle_indices = np.array(list(itertools.combinations(range(n_atoms), 3)))

    a1 = bond_angles(conf['XYZList'], angle_indices)
    a2 = __bond_angles(conf['XYZList'], angle_indices)
    eq(a1, a2)