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

from Emsmbuilder import FahProject
from Emsmbuilder import CreateMergedTrajectoriesFromFAH # TJL: We will eventually kill this
from Emsmbuilder import Project
try:
    from deap import dtm
except:
    pass

from Emsmbuilder.arglib import ArgumentParser


# TG 25 jul 2011 from http://code.activestate.com/recipes/285264-natural-string-sorting/
def keynat(string):
    r'''A natural sort helper function for sort() and sorted()
    without using regular expression.

    >>> items = ('Z', 'a', '10', '1', '9')
    >>> sorted(items)
    ['1', '10', '9', 'Z', 'a']
    >>> sorted(items, key=keynat)
    ['1', '9', '10', 'Z', 'a']
    '''
    r = []
    for c in string:
        try:
            c = int(c)
            try: r[-1] = r[-1] * 10 + c
            except: r.append(c)
        except:
            r.append(c)
    return r


def run(projectfn, PDBfn, InputDir, source, discard, mingen, stride, rmsd_cutoff, 
        parallel='None'):

    # check if we are doing an update or a fresh run
    if os.path.exists( projectfn ):
        print "Found project info file encoding previous work, running in update mode..."
        update = True
    else:
        update = False

    # Check output paths
    if not update:
        if not os.path.exists("Trajectories"):
            print "Making directory 'Trajectories' to contain output."
        else:
            print "The directory 'Trajectories' already exists! Exiting"; sys.exit(1)

        if os.path.exists("ProjectInfo.h5"): 
            print "The file 'ProjectInfo.h5' already exists! Exiting"; sys.exit(1)

    print "Looking for", source, "style data in", InputDir
    
    # Source "gpugrid" is a synonym for 'file_dcd'
    if source == "gpugrid":
        source = "file_dcd"

    # If file option selected, search through that file and pull out all xtc files
    # Source can be file, file_dcd
    if source.startswith('file'):
        print "WARNING: Sorting trajectory files by numerical values in their names."
        print "Ensure that numbering is as intended."
        Flist = []

        if 'dcd' in source: itype='.dcd'
        else: itype='.xtc'

        traj_dirs = glob.glob(InputDir+"/*")
        traj_dirs.sort(key=keynat)

        print "Found", len(traj_dirs), "trajectories."
        for traj_dir in traj_dirs:
            toadd = glob.glob( traj_dir + '/*'+itype )
            toadd.sort(key=keynat)
            if toadd:
              Flist.append(toadd)
            
        print "Loading data:"
        print Flist
        CreateMergedTrajectoriesFromFAH.CreateMergedTrajectories(PDBfn, Flist, 
                           OutFileType=".lh5", InFileType=itype, Stride=stride, parallel=parallel)

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

        fahproject = FahProject.FahProject( PDBfn, project_number=project_number, projectinfo_file=projectfn )

        if update:
            fahproject.retrieve.update_trajectories()
        else:
            fahproject.retrieve.write_all_trajectories( InputDir, output_dir, stride, rmsd_cutoff, discard, mingen,
                                                        center_conformations, num_proc, input_style )

    else:
        print "Error parsing arguments: source must be either FAH, file or file_dcd.  You entered: ", source
        sys.exit(1)

    print "Checking for project file..."
    if not os.path.exists(projectfn):
        P1=Project.CreateProjectFromDir(Filename=projectfn, ConfFilename=PDBfn,TrajFileType=".lh5")
        print "Created project file:", projectfn
    else: 
        if source == 'file':
            print "Project file: %s already exists! Check to ensure it is what you want." % projectfn
        else:
            print "Project file generated from FAH conversion."

    print "Data setup properly - done."
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
     calculations.""")

    parser.add_argument('project', type=str, description='''The ProjectInfo (.h5) file
        that contains a mapping of the previous work done. If you specify a file that 
        exists on disk, conversion will pick up where it left off and simply add data
        to what has already been done. If you specify a file that doesn't exist, the
        conversion starts from the beginning and writes a ProjectInfo file with that name
        at the end of the conversion. NOTE: If you specify a ProjectInfo file, the conversion
        automatically retrieves the conversion parameters you were using before and uses
        them - all other options you specify will be IGNORED.''')
    parser.add_argument('pdb')
    parser.add_argument('input_dir', description='''Path to the parent directory
        containing subdirectories with MD (.xtc) data. See the description above
        for the appropriate formatting for directory architecture.''')
    parser.add_argument('source', description='''Data source: "file", "file_dcd" or
        "fah". This is the style of trajectory data that gets fed into MSMBuilder.
        If a file, then it requires each trajectory be housed in a separate directory
        like (PROJECT/TRAJ*/frame*.xtc). If 'fah', then standard FAH-style directory
        architecture is required.''', default='file', choices=['fah', 'file'])
    parser.add_argument('discard', description='''Number of frames to discard from the
        end of XTC files. MSMb2 will disregard the last x frames from each XTC.
        NOTE: This has changed from MSMb1.''', type=int, default=0)
    parser.add_argument('mingen', description='''Minimum number of XTC frames 
        required to include data in Project.  Used to discard extremely short 
        trajectories.  Only allowed in conjunction with source = 'FAH'.''',
        default=0, type=int)
    parser.add_argument('stride', description='''Integer number to subsample by.
        Every "u-th" frame will be taken from the data and converted to msmbuilder
        format''', default=1, type=int)
    parser.add_argument('rmsd_cutoff', description='''A safe-guard that discards any
        structure with and RMSD higher than the specified value (in nanometers,
        with respect to the input PDB file). Pass -1 to disable this feature''',
        default=-1, type=float)
    parser.add_argument('parallel', description='''Run the conversion in parallel.
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
    
    if args.parallel == 'dtm':
        dtm.start(run, args.project, args.pdb, args.input_dir, args.source, args.discard,
            args.mingen, args.stride, rmsd_cutoff, args.parallel)
    else:
        run(args.project, args.pdb, args.input_dir, args.source, args.discard,
            args.mingen, args.stride, rmsd_cutoff, args.parallel)
