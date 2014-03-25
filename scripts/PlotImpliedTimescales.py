#!/usr/bin/env python
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
import numpy as np
from msmbuilder import arglib

parser = arglib.ArgumentParser(description="""
\nPlots data generated from CalculateImpliedTimescales.py. You may want to use
this as a template for a pylab session

We recommend modifying this script for your own purposes""")
parser.add_argument('input', help='Path to ImpledTimescales.dat',
                    default='ImpliedTimescales.dat')
parser.add_argument('dt', help='Time between snapshots in your data',
                    default=1, type=float)
parser.add_argument(
    'filename', help='Filename to save plot to. Leave blank to render plot to sceen', default='')
parser.add_argument('title', help='Title for plot',
                    default='Relaxation Timescale versus Lagtime')

def entry_point():
    import matplotlib.pyplot as pp
    args = parser.parse_args()
    input = np.loadtxt(args.input)

    pp.scatter(input[:, 0] * args.dt, input[:, 1] * args.dt)
    pp.yscale('log')
    pp.title(args.title)
    pp.xlabel('Lag Time')
    pp.ylabel('Relaxation Timescale')

    if args.filename:
        pp.savefig(args.filename)
    else:
        pp.show()


if __name__ == "__main__":
    entry_point()
