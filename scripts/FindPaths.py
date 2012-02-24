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

#TJL 2011, PANDE GROUP

import sys
import numpy as np
import os
import scipy.io

from Emsmbuilder import TransitionPathTheory
from Emsmbuilder import Serializer

import ArgLib

def run(NFlux, A, B, n, output):

    # Check output isn't taken
    if os.path.exists(output):
        print "Error: File %s already exists! Exiting." % output
        sys.exit(1)

    (Paths, Bottlenecks, Fluxes) = TransitionPathTheory.DijkstraTopPaths(A, B, NFlux, NumPaths=n, NodeWipe=False)

    # We have to pad the paths with -1s to make a square array
    maxi = 0 # the maximum path length
    for path in Paths:
        if len(path) > maxi: maxi = len(path)
    PaddedPaths = -1 * np.ones( (len(Paths), maxi ) )
    for i, path in enumerate(Paths):
        PaddedPaths[i,:len(path)] = np.array(path)
    
    print "Saving %s..." % output
    fdata = Serializer.Serializer( {'Paths': PaddedPaths,
                                    'Bottlenecks': np.array(Bottlenecks),
                                    'Fluxes': np.array(Fluxes)} )
    fdata.SaveToHDF(output)

    return

if __name__ == "__main__":
    print """\nFinds the highest flux paths through an MSM.
    Returns: an HDF5 file (default: Paths.h5), which contains three items:
    (1) The highest flux pathways (a list of ints)
    (2) The bottlenecks in these pathways (a list of 2-tuples)
    (3) The flux of each pathway
    \nPaths.h5 can be read by RenderPaths.py which generates a .dot file capturing these paths.
    """
    arglist=["number", "tmat", "starting", "ending", "output"]
    options=ArgLib.parse(arglist, Custom=[
        ("tmat", "Net Flux matrix from DoTPT.py. Default: NFlux.mtx", "NFlux.mtx"),
        ("number", "The number of pathways you want to retrieve.", None),
        ("output", "Name of the output file. Default: Paths.h5", "NoOutputSet")])
    print sys.argv

    if options.output == 'NoOutputSet': output = "Paths.h5"
    else: output = options.output

    n=int(options.number)
    NFlux=scipy.io.mmread(str(options.tmat))

    T = scipy.io.mmread(str(options.tmat))
    F = np.loadtxt(options.ending, int)
    U = np.loadtxt(options.starting, int)

    # deal with case where have single start or end state
    if F.shape == ():
        tmp = np.zeros(1, dtype=int)
        tmp[0] = int(F)
        F = tmp.copy()
    if U.shape == ():
        tmp = np.zeros(1, dtype=int)
        tmp[0] = int(U)
        U = tmp.copy()

    run(NFlux, U, F, n, output)
