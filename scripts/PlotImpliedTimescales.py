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

import ArgLib

print """
\nPlots data generated from CalculateImpliedTimescales.py. You may want to use
this as a template for a pylab session

*** THIS IS A TEMPLATE ONLY ***\n"""

arglist=["input", "dt"]
options=ArgLib.parse(arglist)
print sys.argv

lagTimes = loadtxt(options.input)
dt = float(options.dt)

scatter(lagTimes[:,0]*dt, lagTimes[:,1]*dt)
yscale('log')
title('Relaxation Timescales versus Lagtime')
xlabel('Lag Time')
ylabel('Relaxation Timescale')

show()
