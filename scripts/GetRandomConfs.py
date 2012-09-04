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

import sys
import os

from msmbuilder import Project
from msmbuilder import Serializer
from msmbuilder import arglib
import logging
logger = logging.getLogger(__file__)

def run(project, assignments, num_confs_per_state, output, format):
    arglib.die_if_path_exists(output)
    num_states = max(assignments.flatten()) + 1
    logger.info("Pulling %s confs for each of %s confs", num_confs_per_state, num_states)
    
    random_confs = project.GetRandomConfsFromEachState(assignments, num_states, num_confs_per_state)
    
    if format == 'pdb':
        random_confs.SaveToPDB(output)
    elif format == 'lh5':
        random_confs.SaveToLHDF(output)
    elif format == 'xtc':
        random_confs.SaveToXTC(output)
    else:
        raise ValueError('Unrecognized format')
   
    logger.info("Saved output to %s", output)
    


if __name__ == "__main__":
    parser = arglib.ArgumentParser(description="""
Pulls a certain number of random conformations from each cluster. Returns these
as an HDF5/PDB/XTC file that contains one long chain of these conformations that looks
like a Trajectory. If you selected to sample N conformations from each cluster,
the first N conformations are from cluster 0, the next N from cluster 1, etc.

Output default: XRandomConfs.lh5, where X=Number of Conformations.""")
    parser.add_argument('project')
    parser.add_argument('assignments', default='Data/Assignments.Fixed.h5')
    parser.add_argument('output', help="""The name of the RandomConfs
        trajectory (.lh5) to write. XRandomConfs.lh5, where X=Number of
        Conformations.""", default='XRandomConfs')
    parser.add_argument('conformations_per_state', help='''Number of
        conformations to randomly sample from your data per state''', type=int)
    parser.add_argument('format', help='''Format to output the data in. Note
        that the PDB format is uncompressed and not efficient. For XTC, you can view
        the trajectory using your project's topology file''', default='lh5',
        choices=['pdb', 'xtc', 'lh5'])    
    args = parser.parse_args()
    
    if args.output == 'XRandomConfs':
            args.output = '%dRandomConfs.%s' % (args.conformations_per_state, args.format)

    run(args.project, args.assignments['Data'], args.conformations_per_state,
        args.output, args.format)
