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
import os

from msmbuilder.project import validators, ProjectBuilder, FahProjectBuilder
from msmbuilder.arglib import ArgumentParser, die_if_path_exists
import logging
logger = logging.getLogger(__name__)


def run(projectfn, PDBfn, InputDir, source, min_length, stride, rmsd_cutoff):
    
    # check if we are doing an update or a fresh run
    if os.path.exists(projectfn):
        logger.info("Found project info file encoding previous work, running in update mode...")
        update = True
    else:
        update = False
    
    logger.info("Looking for %s stype data in %s", source, InputDir)
    if update:
        raise NotImplementedError("Ack! Update mode is not yet ready yet.")
    
    
    # if the source is fah, we'll use some special FaH specific loading functions
    # to (1) try to recover in case of errors and (2) load the specific directory
    # hierarchy of FaH (RUN/CLONE/GEN/frame.xtc)
    if source.startswith('file'):
        itype = '.dcd' if 'dcd' in source else '.xtc'
        pb = ProjectBuilder(InputDir, input_traj_ext=itype, conf_filename=PDBfn, stride=stride)
    elif source == 'fah':
        pb = FahProjectBuilder(InputDir, input_traj_ext='.xtc', conf_filename=PDBfn, stride=stride)
    else:
        raise ValueError("Invalid argument for source: %s" % source)

    
    # check that trajectories to not go farther than a certain RMSD
    # from the PDB. Useful to check for blowing up or other numerical instabilities
    if rmsd_cutoff is not None:
        # TODO: this is going to use ALL of the atom_indices, including hydrogen. This is
        # probably not the desired functionality
        validator = validators.RMSDExplosionValidator(PDBfn, max_rmsd=rmsd_cutoff, atom_indices=None)
        pb.add_validator(validator)
    
    # Only accept trajectories with more snapshots than min_length.
    if min_length > 0:
        validator = validators.MinLengthValidator(min_length)
        pb.add_validator(validator)
    
    # everyone wants to be centered
    pb.add_validator(validators.TrajCenterer())
    
    pb.get_project().save(projectfn)
    assert os.path.exists(projectfn), '%s does not exist' % projectfn
    logger.info("Finished data conversion successfully.")
    logger.info("Generated: %s, Trajectories/", projectfn)
    
    return


if __name__ == "__main__":
    parser = ArgumentParser(description="""
Merges individual XTC files into continuous lossy HDF5 (.lh5) trajectories.

Can read data from a FAH project (PROJECT/RUN*/CLONE*/frame*.xtc) or from
a directory containing one directory for each trajectory, with all the
relevant XTCs inside that directory (PROJECT/TRAJ*/frame*.xtc).

Output:
  -- 'Trajectories' directory containing all the merged lh5 files
  -- 'ProjectInfo' file containing information MSMBuilder uses in subsequent
     calculations.

NOTE: There has been a change from previous versions of MSMBuilder with
regards to the way snapshots are discarded. In the 'FAH' style reader,
there is an automatic check to see if any two snapshots are the same at
the beginning/end of consecutive xtc/dcd files. If they are the same, one
gets discarded. Further, the FahProject object retains a little more discard
functionality.
""")
    
    parser.add_argument('project', type=str, help='''The ProjectInfo (.h5) file
        that contains a mapping of the previous work done. If you specify a file that
        exists on disk, conversion will pick up where it left off and simply add data
        to what has already been done. If you specify a file that doesn't exist, the
        conversion starts from the beginning and writes a ProjectInfo file with that name
        at the end of the conversion. NOTE: If you specify a ProjectInfo file, the conversion
        automatically retrieves the conversion parameters you were using before and uses
        them - all other options you specify will be IGNORED.''')
    parser.add_argument('pdb')
    parser.add_argument('input_dir', help='''Path to the parent directory
        containing subdirectories with MD (.xtc/.dcd) data. See the description above
        for the appropriate formatting for directory architecture.''')
    parser.add_argument('source', help='''Data source: "file", "file_dcd" or
        "fah". This is the style of trajectory data that gets fed into MSMBuilder.
        If a file, then it requires each trajectory be housed in a separate directory
        like (PROJECT/TRAJ*/frame*.xtc). If 'fah', then standard FAH-style directory
        architecture is required.''', default='file', choices=['fah', 'file', 'file_dcd'])
    parser.add_argument('min_length', help='''Minimum number of frames per trajectory
        required to include data in Project.  Used to discard extremely short
        trajectories.''', default=0, type=int)
    parser.add_argument('stride', help='''Integer number to subsample by.
        Every "u-th" frame will be taken from the data and converted to msmbuilder
        format''', default=1, type=int)
    parser.add_argument('rmsd_cutoff', help='''A safe-guard that discards any
        structure with and RMSD higher than the specified value (in nanometers,
        with respect to the input PDB file). Pass -1 to disable this feature''',
        default=-1, type=float)
    #parser.add_argument('parallel', help='''Run the conversion in parallel.
    #    multiprocessing launches multiple python interpreters to use all of your cores.
    #    dtm uses mpi, and requires python's "deap" module to be installed. To execute the
    #    code over mpi using dtm, you need to start the command with mpirun -np <num_procs>.
    #    Note that in many circumstates, the conversion done by this script is IO bound,
    #    not CPU bound, so parallelism can actually be detrememtal.''', default='None',
    #    choices=['None', 'multiprocessing', 'dtm'])
    args = parser.parse_args()
    
    rmsd_cutoff = args.rmsd_cutoff
    if rmsd_cutoff <= 0.0:
        rmsd_cutoff = None
    else:
        logger.warning("Will discard any frame that is %f nm from the PDB conformation...", rmsd_cutoff)
    
    run(args.project, args.pdb, args.input_dir, args.source,
        args.min_length, args.stride, rmsd_cutoff)
