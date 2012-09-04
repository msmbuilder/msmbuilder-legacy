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
import re
import glob
import cPickle

from msmbuilder import FahProject
from msmbuilder import Project
from msmbuilder.utils import keynat

try:
    from deap import dtm
except:
    pass

from msmbuilder.arglib import ArgumentParser, die_if_path_exists


def yield_trajectory_filelist_from_dir( InputDir, itype ):

    print "\nWARNING: Sorting trajectory files by numerical values in their names."
    print "Ensure that numbering is as intended."

    traj_dirs = glob.glob(InputDir+"/*")
    traj_dirs.sort(key=keynat)

    Flist = [] # will hold a list of all the files

    print "\nFound", len(traj_dirs), "trajectories."
    for traj_dir in traj_dirs:
         toadd = glob.glob( traj_dir + '/*'+itype )
         toadd.sort(key=keynat)
         if toadd:
             Flist.append(toadd)

    print "\nLoading data:"
    print Flist

    return Flist


def run(projectfn, PDBfn, InputDir, source, mingen, stride, rmsd_cutoff, 
        parallel='None'):

    # check if we are doing an update or a fresh run
    if os.path.exists( projectfn ):
        print "Found project info file encoding previous work, running in update mode..."
        update = True
    else:
        update = False

    # Check output paths
    if not update:
        die_if_path_exists("Trajectories")
        die_if_path_exists("ProjectInfo.h5")

    print "Looking for", source, "style data in", InputDir
    
    # Source "gpugrid" is a synonym for 'file_dcd'
    if source == "gpugrid":
        source = "file_dcd"

    # If file option selected, search through that file and pull out all xtc files
    # Source can be file, file_dcd
    if source.startswith('file'):

        if update:
            raise NotImplementedError("Ack! Update mode is not yet ready for 'file' mode")
        

        if 'dcd' in source: itype='.dcd'
        else: itype='.xtc'

        Flist = yield_trajectory_filelist_from_dir( InputDir, itype )
        Project.convert_trajectories_to_lh5(PDBfn, Flist,
                                            input_file_type=itype, stride=stride, 
                                            parallel=parallel)

        # memory is a map: memory[ trajectory-dir ] = (lh5-file-path, num-files-in-traj-dir)
        memory = {}
        for i, trj_dir in enumerate( os.listdir( InputDir ) ): # TJL: should check this!!
            num_raw_trajs = len( glob.glob( trj_dir + "/*" + itype ) )
            memory[ trj_dir ] = ( "trj%d.lh5" % i, num_raw_trajs )

        Project.CreateProjectFromDir(Filename=projectfn, ConfFilename=PDBfn, 
                                     TrajFileType=".lh5", initial_memory=cPickle.dumps( memory ))

    # If FAH option, seach through a RUNs/CLONEs/frameS.xtc data structure
    elif source == 'fah':

        # hard-wiring some arguments, these could be revealed but probably aren't necessary
        output_dir = "./Trajectories"
        center_conformations = True
        input_style = source.upper()
        try:
            project_number = re.match('\w+(\d+)\w+', InputDir).group()
            print "Converting FAH Project %d" % project_number
        except:
            project_number = 0 # this number is not critical
        
        # check parallelism mode, and set the number of processors accordingly 
        if parallel == 'multiprocessing':
            num_proc = int(os.sysconf('SC_NPROCESSORS_ONLN'))
            print "Found and using %d processors in parallel" % num_proc
        elif parallel == 'None':
            num_proc = 1
        else:
            print "Allowed parallel options for FAH: None or multiprocessing"
            raise Exception("Error parsing parallel option: %s" % parallel)

        fahproject = FahProject( PDBfn, project_number=project_number, projectinfo_file=projectfn )

        if update:
            fahproject.retrieve.update_trajectories()
        else:
            fahproject.retrieve.write_all_trajectories( InputDir, output_dir, stride, rmsd_cutoff, mingen,
                                                        center_conformations, num_proc, input_style )

    else:
        raise Exception("Invalid argument for source: %s" % source)

    assert os.path.exists(projectfn)
    print "\nFinished data conversion successfully."
    print "Generated: %s, Trajectories/, Data/" % projectfn

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
        containing subdirectories with MD (.xtc) data. See the description above
        for the appropriate formatting for directory architecture.''')
    parser.add_argument('source', help='''Data source: "file", "file_dcd" or
        "fah". This is the style of trajectory data that gets fed into MSMBuilder.
        If a file, then it requires each trajectory be housed in a separate directory
        like (PROJECT/TRAJ*/frame*.xtc). If 'fah', then standard FAH-style directory
        architecture is required.''', default='file', choices=['fah', 'file', 'file_dcd'])
    parser.add_argument('mingen', help='''Minimum number of XTC frames 
        required to include data in Project.  Used to discard extremely short 
        trajectories.  Only allowed in conjunction with source = 'FAH'.''',
        default=0, type=int)
    parser.add_argument('stride', help='''Integer number to subsample by.
        Every "u-th" frame will be taken from the data and converted to msmbuilder
        format''', default=1, type=int)
    parser.add_argument('rmsd_cutoff', help='''A safe-guard that discards any
        structure with and RMSD higher than the specified value (in nanometers,
        with respect to the input PDB file). Pass -1 to disable this feature''',
        default=-1, type=float)
    parser.add_argument('parallel', help='''Run the conversion in parallel.
        multiprocessing launches multiple python interpreters to use all of your cores.
        dtm uses mpi, and requires python's "deap" module to be installed. To execute the
        code over mpi using dtm, you need to start the command with mpirun -np <num_procs>.
        Note that in many circumstates, the conversion done by this script is IO bound,
        not CPU bound, so parallelism can actually be detrememtal.''', default='None',
        choices=['None', 'multiprocessing', 'dtm'])
    args = parser.parse_args()
    
    rmsd_cutoff = args.rmsd_cutoff
    if rmsd_cutoff<=0.:
        rmsd_cutoff=1000.
    else:
        print "WARNING: Will discard any frame that is %f nm from the PDB conformation..." % rmsd_cutoff
    
    if args.parallel == 'dtm' and args.source != 'file':
        raise NotImplementedError('Sorry. At this point parallelism is only implemented for file-style')
    
    run(args.project, args.pdb, args.input_dir, args.source,
        args.mingen, args.stride, rmsd_cutoff, args.parallel)
