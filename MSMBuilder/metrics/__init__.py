"""
This should be documented better somewhere, because it will cause cryptic
errors if you don't do it. Whatever data structure you return from
prepare_trajectory() needs to support slice sytax. If you return an array or
something, then this is no problem, but if you create your own object to hold
the data that prepare_trajectory() returns, you need to add a __getitem__(),
__setitem__() and __len__() methods. See the RMSD.TheoData object for an
example. Also, if you're not familiar with this side of python these docs
(http://docs.python.org/reference/datamodel.html#emulating-container-types)
are pretty good. Only __getitem__, __setitem__ and __len__ are necessary.
"""
from __future__ import print_function, division, absolute_import
from .baseclasses import AbstractDistanceMetric
from .baseclasses import Vectorized
from .rmsd import RMSD
from .dihedral import Dihedral
from .contact import BooleanContact, AtomPairs, ContinuousContact
from .projection import RedDimPNorm
from .hybrid import Hybrid, HybridPNorm
from .projection import RedDimPNorm
from .positions import Positions
from .solvent import SolventFp
