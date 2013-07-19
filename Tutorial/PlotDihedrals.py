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
"""Creates Ramachandran plot from raw dipeptide data and macroscopic MSM."""
from pylab import *
from numpy import loadtxt
from matplotlib import *
from matplotlib.pyplot import *
import sys
import msmbuilder.io


if len(sys.argv) != 2:
    print("Usage: python PlotDihedrals.py Filename")
    print("Where Filename is the location of the Dihedrals.h5 data file.")
    sys.exit(0)
Filename=sys.argv[1]

#Load data
dihedral_data = msmbuilder.io.loadh(Filename)
phi = dihedral_data['Phi']
psi = dihedral_data['Psi']
ind = dihedral_data['StateIndex']
NumStates=len(ind)

# Helper function to give the indices for a particular macrostate i
def w(i, ind):
    prev = sum(ind[:i])
    return range(prev, prev+ind[i])

for i in xrange(NumStates):
    plot(phi[w(i,ind)], psi[w(i,ind)], "x", label="State %d"%i)
    axis([-180,180,-180,180])


LabelList=[",",".",'o',"<","s","*","h","+","D"]
ColorList=["b","g","r","c","m","y","k"]
title("Alanine Dipeptide Macrostates")
xlabel(r"$\phi$")
ylabel(r"$\psi$")
axis([-180,180,-180,180])
import matplotlib.font_manager
prop = matplotlib.font_manager.FontProperties(size=8.0)
legend(loc=0,labelspacing=0.075,prop=prop,scatterpoints=1,markerscale=0.5,numpoints=1)

plot([-180,0],[50,50],'k')
plot([-180,0],[-100,-100],'k')
plot([0,180], [100,100],'k')
plot([0,180], [-50,-50],'k')
plot([0,0],[-180,180],'k')
plot([-100,-100],[50,180],'k')
plot([-100,-100],[-180,-100],'k')


annotate("PPII",[-30,150],fontsize='x-large')
annotate(r"$\beta$",[-130,150],fontsize='x-large')
annotate(r"$\alpha$",[-100,0],fontsize='x-large')
annotate(r"$\alpha_L$",[100,30],fontsize='x-large')
annotate(r"$\gamma$",[100,-150],fontsize='x-large')
show()

