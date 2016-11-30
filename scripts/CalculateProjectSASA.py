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

import numpy as np
import mdtraj as md
from mdtraj import io
from msmbuilder import Project, arglib
import os
import logging
logger = logging.getLogger('msmbuilder.scripts.CalculateProjectSASA')

parser = arglib.ArgumentParser(description="""Calculates the Solvent Accessible Surface Area
    of all atoms in a subset of the trajectories, or for all trajectories in the project. The
    output is a hdf5 file which contains the SASA for each atom in each frame
    in each trajectory (or the single trajectory you passed in.""" )
parser.add_argument('project')
parser.add_argument('outdir', help=r"""Output directory to save .h5 files for each trajectory.
    Each will be named sasa<index>.h5 where <index> corresponds to the project's index for that 
    trajectory""", default='sasa')
parser.add_argument('which', help="""which trajectories to calculate the SASA for.
    This script saves a separate file for each trajectory.""", default=[0, np.inf],
    nargs=2, type=int)

def run(project, outdir, which):

    which[0] = np.max([0, which[0]])
    which[1] = np.min([project.n_trajs, which[1]])

    n_atoms = project.load_conf().n_atoms

    for traj_ind in xrange(*which):
        logger.info("Working on Trajectory %d", traj_ind)
        traj_asa = []
        out_fn = os.path.join(args.outdir, 'sasa%d.h5' % traj_ind)
        if os.path.exists(out_fn):
            logger.warn("!!!!! Path (%s) exists, so I am skipping this trajectory. Remove %s to recalculate" % (out_fn, out_fn))
            continue

        traj_fn = project.traj_filename(traj_ind)
        for traj_chunk in md.iterload(traj_fn, chunk=1000):
            traj_asa.extend(md.shrake_rupley(traj_chunk))

        traj_asa = np.array(traj_asa)
        io.saveh(out_fn, traj_asa)

    return 


def entry_point():
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    if not os.path.isdir(args.outdir):
        raise Exception("output directory (%s) exists but is not a directory!" % args.outdir)

    project = Project.load_from(args.project)
    run(project, args.outdir, args.which)


if __name__ == "__main__":
    entry_point()

