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

from msmbuilder.project import validators, ProjectBuilder, FahProjectBuilder
from msmbuilder import Project
from msmbuilder.arglib import ArgumentParser, die_if_path_exists
from mdtraj.utils import ensure_type

logger = logging.getLogger('msmbuilder.scripts.ConvertDataToHDF')


parser = ArgumentParser(description="""
Merges individual XTC files into continuous HDF5 (.h5) trajectories.

Can read data from a FAH project (PROJECT/RUN*/CLONE*/frame*.xtc) or from
a directory containing one directory for each trajectory, with all the
relevant XTCs inside that directory (PROJECT/TRAJ*/frame*.xtc).

Output:
-- 'Trajectories' directory containing all the merged h5 files
-- 'ProjectInfo' file containing information MSMBuilder uses in subsequent
 calculations.

NOTE: There has been a change from previous versions of MSMBuilder with
regards to the way snapshots are discarded. In the 'FAH' style reader,
there is an automatic check to see if any two snapshots are the same at
the beginning/end of consecutive xtc/dcd files. If they are the same, one
gets discarded. Further, the FahProject object retains a little more discard
functionality.
""")
parser.add_argument('project', type=str, help='''The ProjectInfo (.h5) to
    write to disk. Contains metadata associated with your project''')
parser.add_argument('pdb')
parser.add_argument('input_dir', help='''Path to the parent directory
    containing subdirectories with MD (.xtc/.dcd) data. See the description above
    for the appropriate formatting for directory architecture.''')
parser.add_argument('source', help='''Data source: "file", or
    "fah". For "file" format, each of the trajectories needs to be
    in a different directory. For example, if you supply input_dir='XTC', then
    it is expected that the directory 'XTC' contains a set of subdirectories, each
    of which contains one or more files of a single MD trajectory that will be concatenated
    together. The glob pattern used would be XTC/*/*.xtc'. If 'fah', then standard
    folding@home-style directory architecture is required.''',
                    default='file', choices=['fah', 'file'])
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
parser.add_argument('atom_indices', help='''If specified, load atom indices
    using np.loadtxt() and pass along to the converter.  This allows you to 
    extract only a subset of atoms during the file conversion process.''',
                    default="", type=str)                    
parser.add_argument('iext', help='''The file extension of input trajectory
    files.  Must be a filetype that mdtraj.load() can recognize.''',
                    default=".xtc", type=str)

def run(projectfn, conf_filename, input_dir, source, min_length, stride, rmsd_cutoff, atom_indices, iext):

    # check if we are doing an update or a fresh run
    # if os.path.exists(projectfn):
    #     logger.info("Found project info file encoding previous work, running in update mode...")
    #     update = True
    # else:
    #     update = False
    #
    # logger.info("Looking for %s style data in %s", source, input_dir)
    # if update:
    #     raise NotImplementedError("Ack! Update mode is not yet ready yet.")

    # if the source is fah, we'll use some special FaH specific loading functions
    # to (1) try to recover in case of errors and (2) load the specific directory
    # hierarchy of FaH (RUN/CLONE/GEN/frame.xtc)
    if os.path.exists(projectfn):
        project = Project.load_from(projectfn)
        logger.warn(
            "%s exists, will modify it and update the trajectories in %s",
            projectfn, '/'.join(project._traj_paths[0].split('/')[:-1]))
    else:
        project = None

    if source.startswith('file'):
        pb = ProjectBuilder(
            input_dir, input_traj_ext=iext, conf_filename=conf_filename,
            stride=stride, project=project, atom_indices=atom_indices)
    elif source == 'fah':
        pb = FahProjectBuilder(
            input_dir, input_traj_ext=iext, conf_filename=conf_filename,
            stride=stride, project=project, atom_indices=atom_indices)
    else:
        raise ValueError("Invalid argument for source: %s" % source)

    # check that trajectories to not go farther than a certain RMSD
    # from the PDB. Useful to check for blowing up or other numerical
    # instabilities
    if rmsd_cutoff is not None:
        # TODO: this is going to use ALL of the atom_indices, including hydrogen. This is
        # probably not the desired functionality
        # KAB: Apparently needed to use correctly subsetted atom_indices here to avoid an error
        validator = validators.RMSDExplosionValidator(
            conf_filename, max_rmsd=rmsd_cutoff, atom_indices=atom_indices)
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
    args = parser.parse_args()
    rmsd_cutoff = args.rmsd_cutoff
    if rmsd_cutoff <= 0.0:
        rmsd_cutoff = None
    else:
        logger.warning(
            "Will discard any frame that is %f nm from the PDB conformation...", rmsd_cutoff)

    if args.atom_indices == "":
        atom_indices = None
    else:
        atom_indices = np.loadtxt(args.atom_indices)
        atom_indices = ensure_type(atom_indices, 'int', 1, "atom_indices")

    run(args.project, args.pdb, args.input_dir, args.source,
        args.min_length, args.stride, rmsd_cutoff, atom_indices, args.iext)
