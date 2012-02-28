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
import numpy
import scipy.io

from msmbuilder import TransitionPathTheory

import ArgLib

def run(TC, Uv, Fv, Populations):

    # Check output isn't taken
    output_flist = ["BCommittors.dat", "FCommittors.dat", "NFlux.mtx"]
    for output in output_flist:
        if os.path.exists(output):
            print "Error: File %s already exists! Exiting." % output
            sys.exit(1)

    # Get committors and flux
    print "Getting committors and flux..."
    Bc = TransitionPathTheory.GetBCommittors(Uv, Fv, TC, Populations, maxiter=100000, X0=None, Dense=False)
    print "Calculated reverse committors."
    Fc = TransitionPathTheory.GetFCommittors(Uv, Fv, TC, maxiter=100000, X0=None, Dense=False)
    print "Calculated forward committors."
    NFlux = TransitionPathTheory.GetNetFlux(Populations, Fc, Bc, TC)
    print "Calculated net flux."

    numpy.savetxt("BCommittors.dat", Bc)
    numpy.savetxt("FCommittors.dat", Fc)
    scipy.io.mmwrite("NFlux.mtx", NFlux)
    print "Wrote: BCommittors.dat, FCommittors.dat, NFlux.mtx"

if __name__ == "__main__":
    print """\nCalculates a number of kinetic transition properties of a given MSM. Returns:
    (1) FCommittors.dat - the forward committors of the MSM (numpy savetxt)
    (2) BCommittors.dat - the backward committors (numpy savetxt)
    (3) NFlux.mtx - the net flux matrix (scipy sparse fmt)\n
    """
    arglist=["tmat", "starting", "ending", "populations"]
    options=ArgLib.parse(arglist)
    print sys.argv

    T = scipy.io.mmread(str(options.tmat))
    U = numpy.loadtxt(options.starting, int)
    F = numpy.loadtxt(options.ending, int)
    Pops = numpy.loadtxt(options.populations)

    # deal with case where have single start or end state
    if U.shape == ():
        tmp = numpy.zeros(1, dtype=int)
        tmp[0] = int(U)
        U = tmp.copy()
    if F.shape == ():
        tmp = numpy.zeros(1, dtype=int)
        tmp[0] = int(F)
        F = tmp.copy()

    run(T, U, F, Pops)
