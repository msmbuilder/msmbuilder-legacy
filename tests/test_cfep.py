
"""
This is TJL's slightly modified version of Sergei Krivov's original python
code to calculate min-cut max-flow reaction coordinates. TJL basically
just made it compatible with MSMBuidler by employing numpy/scipy.

To Do
-----
Test all code: need generators for WW domain

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import nose
import numpy as np
import scipy
from scipy import io
from mdtraj.utils.six import PY3

from msmbuilder import cfep
from msmbuilder.testing import get

def ref_calc_cFEP(counts, lag_time, rxn_coordinate, rescale=True):
    """
    Computes the partition function of the cut-based free energy profile based
    on transition network and reaction coordinate. 
    
    Employs an MSM to do so, thus taking input in the form of a discrete state
    space and the observed dynamics in that space (assignments). Further, one
    can provide a "reaction coordiante" who's values will be employed to
    find the min-cut/max-flow coordinate describing dynamics on that space.
    
    Generously contributed by Sergei Krivov
    Optimizations and modifications due to TJL
    """

    counts = counts.tolil()

    # initialize data structures
    zc = {} # cut-based partition function (1/2 number of transitions)
    zh = {} # histogram-based partition function (number seen / bin width)

    # find the counts
    elements = len(counts.nonzero()[0])
    N = counts.shape[0]
    
    # loop over all elements in the counts matrix
    for x,(i,j) in enumerate( np.array( counts.nonzero() ).T ):                
        nij = counts[i,j]
        xi = rxn_coordinate[i]
        xj = rxn_coordinate[j]
        zh[xj] = zh.get(xj, 0) + nij
        if xi > xj:
            zc[xj] = zc.get(xj, 0) + nij
            zc[xi] = zc.get(xi, 0) - nij
        elif xj > xi:
            zc[xi] = zc.get(xi, 0) + nij
            zc[xj] = zc.get(xj, 0) - nij

    zc_keys = list(zc.keys())
    zc_keys.sort()

    # If no reaction coordinate is provided, calculate the "natural"
    # coordinate where D(x) = 1 and cfep(x) = hfep(x)
    if rescale:
        print("rescaling to the natural coordinate")
        szc = 0
        sx  = 0
        x2nx = {}
        zcn  = {}
        for x in zc_keys:
            szc = szc + zc[x] / 2.0
            if x in zh and szc > 0:
                sx += float(zh[x]) / (szc * math.sqrt(math.pi))
            zcn[sx] = szc
            x2nx[x] = sx

    else:
        for k in list(zc.keys()):
            zc[k] /= 2.0
    
    return zc, zh


class TestCfep():
    """ test the cfep library in msmbuilder by comparing to the reference
        implementation above """
    

    def setUp(self):
        if PY3:
            raise nose.SkipTest('TestCfep requires scipy.weave which doesnt exist on python3')
        
        self.generators = get('cfep_reference/Gens.lh5')
        N = len(self.generators)
        
        self.counts = get('cfep_reference/tCounts.mtx')
        self.lag_time = 1.0
        self.pfolds = np.random.rand(N)
        self.rescale = False
        
        self.reactant = 0
        self.product  = N
        
    
    def test_feps(self):
        
        pfold_cfep = cfep.CutCoordinate(self.counts, self.generators, 
                                        self.reactant, self.product)
        if self.rescale:
            pfold_cfep.rescale_to_natural_coordinate()
        pfold_cfep.set_coordinate_values(self.pfolds)
        
        zc = pfold_cfep.zc
        zh = pfold_cfep.zh
        
        zc_ref, zh_ref = ref_calc_cFEP(self.counts, self.lag_time, self.pfolds, self.rescale)

        error = 0.0
        for k in list(zc_ref.keys()):
            error += np.abs(zc_ref[k] - zc[np.where( self.pfolds == k )].flatten()[0])

        assert np.abs( error ) < 0.0000001

        return



if __name__ == '__main__':
    test()
  
