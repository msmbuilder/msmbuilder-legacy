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

import os
import sys
import numpy as np
from scipy import io

from msmbuilder import MSMLib
from msmbuilder import Serializer

def docstring():
    print """
    Generates the most likely rate matrix given a set of observed data,
    in the form of a counts matrix and assignments file. The assignments
    are used only to get the dwell times in each state. You can pass in a
    modified (e.g. symmetrized) counts matrix or a modified (e.g. ergodicly
    trimmed) assignments file. In fact, you should definitely be doing so!

    Usage: python Estimate-K-from-C.py <Assignments.h5> <tCounts.mtx>
    Output: 'K.mtx', rate matrix in scipy sparse format.
    """
    sys.exit(0)
    return

def parse_error(err_out):
    print "Error parsing arguments: %s" % err_out
    print "Usage: python Estimate-K-from-C.py <Assignments.h5> <tCounts.mtx>"
    sys.exit(1)
    return

def main(Assignments, C):

    if os.path.exists('K.mtx'):
        print "Error! 'K.mtx', file already exists!"
        sys.exit(1)
    K = MSMLib.EstimateRateMatrix(C, Assignments)
    io.mmwrite('K.mtx', K)
    print "Wrote: K.mtx"

    return


if __name__ == '__main__':
    print sys.argv

    if sys.argv[1] == '-h':
        docstring()
    elif sys.argv[1] == '--help':
        docstring()
    elif len(sys.argv) != 3:
        parse_error("Incorrect number of args")

    ass_fn = sys.argv[1]
    if ass_fn[-3:] != '.h5':
        parse_error("Assignments file has no .h5 extension")
    else:
        Assignments = Serializer.LoadData(ass_fn)

    C_fn = sys.argv[2]
    if C_fn[-4:] != '.mtx':
        parse_error("Counts matrix file has no .mtx extension")
    else:
        C = io.mmread(C_fn)

    main(Assignments, C)


