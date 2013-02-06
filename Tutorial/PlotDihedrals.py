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

print("Usage: python PlotDihedrals.py Filename")
print("Filename should be the location of your Macrostate assignments file.")
Filename=sys.argv[1]

#Load assigment, phi, and psi data.
Ass=msmbuilder.io.loadh(Filename, "arr_0")
phi=msmbuilder.io.loadh("./Phi.h5", "Data")
psi=msmbuilder.io.loadh("./Psi.h5", "Data")

NumStates=Ass.max()+1

"""
hexbin(phi.flatten(),psi.flatten())
title("Ramachandran plot of raw alanine dipeptide data.")
xlabel(r"$\phi$")
ylabel(r"$\psi$")
axis([-180,180,-180,180])
figure()
"""

w=lambda x: where(Ass==x)
LabelList=[",",".",'o',"<","s","*","h","+","D"]
ColorList=["b","g","r","c","m","y","k"]
def PlotIthState(i,Separate=False):
    if Separate==True:
        figure()
    plot(phi[w(i)],psi[w(i)],"x",label="State %d"%i)
    axis([-180,180,-180,180])
    print i, w(i)[0].shape

for i in range(NumStates):
    print(i)
    PlotIthState(i)

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

