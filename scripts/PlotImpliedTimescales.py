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

from pylab import *
from numpy import loadtxt
import sys
from msmbuilder import arglib



parser = arglib.ArgumentParser(description="""
\nPlots data generated from CalculateImpliedTimescales.py. You may want to use
this as a template for a pylab session

We recommend modifying this script for your own purposes""")
parser.add_argument('input', description='Path to ImpledTimescales.dat',
    type=arglib.LoadTxtType(dtype=float), default='ImpliedTimescales.dat')
parser.add_argument('dt', description='Time between snapshots in your data',
    default=1, type=float)
args = parser.parse_args()

scatter(args.input[:,0] * args.dt, args.input[:,1] * args.dt)
yscale('log')
title('Relaxation Timescales versus Lagtime')
xlabel('Lag Time')
ylabel('Relaxation Timescale')

show()
