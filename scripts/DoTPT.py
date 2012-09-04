#!/usr/bin/python
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

import sys
import os
import numpy as np
import scipy.io

from msmbuilder.transition_path_theory import GetBCommittors
from msmbuilder.transition_path_theory import GetFCommittors
from msmbuilder.transition_path_theory import GetNetFlux

from msmbuilder import arglib

def run(TC, Uv, Fv, Populations):

    # Get committors and flux
    print "Getting committors and flux..."
    Bc = GetBCommittors(Uv, Fv, TC, Populations, maxiter=100000, X0=None, Dense=False)
    print "Calculated reverse committors."
    Fc = GetFCommittors(Uv, Fv, TC, maxiter=100000, X0=None, Dense=False)
    print "Calculated forward committors."
    NFlux = GetNetFlux(Populations, Fc, Bc, TC)
    print "Calculated net flux."
    
    return Bc, Fc, NFlux

if __name__ == "__main__":
    parser = arglib.ArgumentParser(description=
"""Calculates a number of kinetic transition properties of a given MSM. Returns:
(1) FCommittors.dat - the forward committors of the MSM (numpy savetxt)
(2) BCommittors.dat - the backward committors (numpy savetxt)
(3) NFlux.mtx - the net flux matrix (scipy sparse fmt)""")
    parser.add_argument('tProb')
    parser.add_argument('starting', help='''Vector of states in the
        starting/reactants/unfolded ensemble.''', default='U_states.dat')
    parser.add_argument('ending', help='''Vector of states in the
        ending/products/folded ensemble.''', default='F_states.dat')
    parser.add_argument('populations', help='''State equilibrium populations
        file, in numpy .dat format.''', default='Data/Populations.dat')
    parser.add_argument('output_dir', default='.')
    args = parser.parse_args()
    
    T = scipy.io.mmread( args.tProb )
    U = np.loadtxt( args.starting ).astype(int)
    F = np.loadtxt( args.ending ).astype(int)
    Pops = np.loadtxt( args.populations ).astype(int)

    # deal with case where have single start or end state
    if U.shape == ():
        tmp = np.zeros(1, dtype=int)
        tmp[0] = int(U)
        U = tmp.copy()
    if F.shape == ():
        tmp = np.zeros(1, dtype=int)
        tmp[0] = int(F)
        F = tmp.copy()

    # Check output isn't taken
    output_list = ["BCommittors.dat", "FCommittors.dat", "NFlux.mtx"]
    output_flist = [os.path.join(args.output_dir, f) for f in output_list]
    arglib.die_if_path_exists(output_flist)
    
    Bc, Fc, NFlux = run(T, U, F, Pops)
    
    np.savetxt(output_flist[0], Bc)
    np.savetxt(output_flist[1], Fc)
    scipy.io.mmwrite(output_flist[2], NFlux)
    print "Wrote: %s" % ', '.join(output_flist)
