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


import os
import sys
import logging
import numpy as np
from msmbuilder import Project
import mdtraj as md
from mdtraj import io
from msmbuilder import arglib
logger = logging.getLogger('msmbuilder.scripts.CalculateProjectDistance')


parser = arglib.ArgumentParser(description="""
Calculate the distance between an input PDB and all conformations in your project.
Alternatively, you can limit the distance calculate to a single trajectory by
passing a trajectory filename.
Output as a HDF5 file (load using msmbuilder.io.loadh())""", get_metric=True)
parser.add_argument('pdb')
parser.add_argument('output', help='''Output file name. Output is an
    .h5 file with RMSD entries corresponding to the Assignments.h5 file.''',
                    default='Data/RMSD.h5')
parser.add_argument('project')
parser.add_argument('traj_fn', help='''Pass a trajectory file, to return
    just the distance for a particular trajectory. Pass 'all' to get all
    distances in the project.''', default='all')


def run(project, pdb, metric, traj_fn=None):

    ppdb = metric.prepare_trajectory(pdb)

    if traj_fn == None:
        distances = -1 * \
            np.ones((project.n_trajs, np.max(project.traj_lengths)))

        for i in xrange(project.n_trajs):
            logger.info("Working on Trajectory %d", i)
            ptraj = metric.prepare_trajectory(project.load_traj(i))
            d = metric.one_to_all(ppdb, ptraj, 0)
            distances[i, 0:len(d)] = d
    else:
        traj = md.load(traj_fn, top=pdb)
        ptraj = metric.prepare_trajectory(traj)

        distances = metric.one_to_all(ppdb, ptraj, 0)

    return distances


def entry_point():
    args, metric = parser.parse_args()

    arglib.die_if_path_exists(args.output)

    project = Project.load_from(args.project)
    pdb = md.load(args.pdb)
    if args.traj_fn.lower() == 'all':
        traj_fn = None
    else:
        traj_fn = args.traj_fn

    distances = run(project, pdb, metric, traj_fn)

    io.saveh(args.output, distances)
    logger.info('Saved to %s', args.output)

if __name__ == "__main__":
    entry_point()
