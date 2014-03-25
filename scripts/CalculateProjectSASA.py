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
import logging
logger = logging.getLogger('msmbuilder.scripts.CalculateProjectSASA')

parser = arglib.ArgumentParser(description="""Calculates the Solvent Accessible Surface Area
of all atoms in a given trajectory, or for all trajectories in the project. The
output is a hdf5 file which contains the SASA for each atom in each frame
in each trajectory (or the single trajectory you passed in.""" )
parser.add_argument('project')
parser.add_argument('atom_indices', help='Indices of atoms to calculate SASA',
    default='all')
parser.add_argument('output', help='''hdf5 file for output. Note this will
    be THREE dimensional: ( trajectory, frame, atom ), unless you just ask for
    one trajectory, in which case it will be shape (frame, atom).''',
    default='SASA.h5')
parser.add_argument('traj_fn', help='''Pass a trajectory file if you only
    want to calclate the SASA for a single trajectory''', default='all' )
        

def run(project, atom_indices=None, traj_fn = 'all'):

    n_atoms = project.load_conf().n_atoms

    if traj_fn.lower() == 'all':

        SASA = np.ones((project.n_trajs, np.max(project.traj_lengths), n_atoms)) * -1

        for traj_ind in xrange(project.n_trajs):
            traj_asa = []
            logger.info("Working on Trajectory %d", traj_ind)
            traj_fn = project.traj_filename(traj_ind)
            chunk_ind = 0
            for traj_chunk in md.iterload(traj_fn, atom_indices=atom_indices, chunk=1000):
                traj_asa.extend(md.shrake_rupley(traj_chunk))
                chunk_ind += 1
            SASA[traj_ind, 0:project.traj_lengths[traj_ind]] = traj_asa

    else:
        traj_asa = []
        for traj_chunk in Trajectory.enum_chunks_from_lhdf( traj_fn, AtomIndices=atom_indices ):
            traj_asa.extend( asa.calculate_asa( traj_chunk ) )

        SASA = np.array(traj_asa)

    return SASA

def entry_point():
    args = parser.parse_args()
    arglib.die_if_path_exists(args.output)

    if args.atom_indices.lower() == 'all':
        atom_indices = None
    else:
        atom_indices = np.loadtxt(args.atom_indices).astype(int)

    project = Project.load_from(args.project)
    SASA = run(project, atom_indices, args.traj_fn)
    io.saveh(args.output, SASA)

if __name__ == '__main__':
    entry_point()
