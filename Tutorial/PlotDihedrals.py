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

import sys
import os
import argparse
import operator
import numpy as np
import matplotlib.pyplot as pp
from mdtraj.utils.six.moves import reduce

import mdtraj as md
from mdtraj import io
from msmbuilder import  Project


PSI_INDICES = [6, 8, 14, 16]
PHI_INDICES = [4, 6, 8, 14]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('assignments', default='Macro4/MacroAssignments.h5', help='Path to an assignments file. (default=Macro4/MacroAssignments.h5)')
    parser.add_argument('--project', default='ProjectInfo.yaml', help='Path to ProjectInfo.yaml file. (default=ProjectInfo.yaml)')
    args = parser.parse_args()

    project = Project.load_from(args.project)
    t = reduce(operator.add, (project.load_traj(i) for i in range(project.n_trajs)))

    phi_angles = md.compute_dihedrals(t, [PHI_INDICES]) * 180.0 / np.pi
    psi_angles = md.compute_dihedrals(t, [PSI_INDICES]) * 180.0 / np.pi
    state_index = np.hstack(io.loadh(args.assignments)['arr_0'])

    for i in np.unique(state_index):
        pp.plot(phi_angles[np.where(state_index == i)],
                psi_angles[np.where(state_index == i)],
                'x', label='State %d' % i)


    pp.title("Alanine Dipeptide Macrostates")
    pp.xlabel(r"$\phi$")
    pp.ylabel(r"$\psi$")
    annotate()

    pp.legend(loc=1, labelspacing=0.075, prop={'size': 8.0}, scatterpoints=1,
              markerscale=0.5, numpoints=1)
    pp.xlim([-180, 180])
    pp.ylim([-180, 180])
    pp.show()


def annotate():
    pp.plot([-180,0],[50,50],'k')
    pp.plot([-180,0],[-100,-100],'k')
    pp.plot([0,180], [100,100],'k')
    pp.plot([0,180], [-50,-50],'k')
    pp.plot([0,0],[-180,180],'k')
    pp.plot([-100,-100],[50,180],'k')
    pp.plot([-100,-100],[-180,-100],'k')

    pp.annotate("PPII",[-30,150],fontsize='x-large')
    pp.annotate(r"$\beta$",[-130,150],fontsize='x-large')
    pp.annotate(r"$\alpha$",[-100,0],fontsize='x-large')
    pp.annotate(r"$\alpha_L$",[100,30],fontsize='x-large')
    pp.annotate(r"$\gamma$",[100,-150],fontsize='x-large')


if __name__ == '__main__':
    main()
