import numpy as np
from msmbuilder import _rg_wrap
from msmbuilder.utils import deprecated

def calculate_rg(xyzlist):
    '''
    Compute the Rg for every frame in an xyzlist
    
    If masses are none, then all the atoms are counted equally.
    Otherwise, you can supply a vector of length num_atoms giving
    the mass of each atom
    '''
    
    traj_length, num_atoms, num_dims = xyzlist.shape
    if not num_dims == 3:
        raise ValueError("What did you pass me?")
    if not xyzlist.dtype == np.float32:
        xyzlist = np.float32(xyzlist)
    
    results = np.zeros(traj_length, dtype=np.double)
    _rg_wrap.rg_wrap(xyzlist, results)
    return results


@deprecated(calculate_rg, '2.7')
def Rg():
    pass