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
import glob

from msmbuilder import CreateMergedTrajectoriesFromFAH
from msmbuilder import Project

import ArgLib

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

def run(projectfn, PDBfn, InputDir, source, discard, mingen, stride, rmsd_cutoff):

    # Check output paths
    if not os.path.exists("Trajectories"): print "Making directory 'Trajectories' to contain output."
    else: print "The directory 'Trajectories' already exists! Exiting"; sys.exit(1)

    if os.path.exists("ProjectInfo.h5"): print "The file 'ProjectInfo.h5' already exists! Exiting"; sys.exit(1)

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
                           OutFileType=".lh5", InFileType=itype,Stride=stride)



    # If FAH option, seach through a RUNs/CLONEs/frameS.xtc data structure
    elif source == 'FAH':
        NumRuns=len( glob.glob( InputDir+"/RUN*" ) )
        clone_str = glob.glob( InputDir+"/RUN*" )[0] + "/CLONE*" 
        NumClones=len( glob.glob(clone_str) )
        print "Found %d RUNs and %d CLONEs" % (NumRuns, NumClones)
        CreateMergedTrajectoriesFromFAH.CreateMergedTrajectoriesFromFAH(PDBfn, InputDir,
                           NumRuns, NumClones, InFilenameRoot="frame", OutFilenameRoot="trj", 
                           OutDir="./Trajectories",OutFileType=".lh5",Stride=stride,MinGen=mingen,
                           DiscardFirstN=discard,trjcatFlags=["-cat"],ProjectFilename=projectfn,
                                                                        MaxRMSD=rmsd_cutoff)

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
        else: print "Project file generated from FAH conversion."

    print "Data setup properly - done."
    return


if __name__ == "__main__":
    print """
\nMerges individual XTC files into continuous lossy HDF5 (.lh5) trajectories. Can
take either data from a FAH project (PROJECT/RUN*/CLONE*/frame*.xtc) or from a
directory containing one directory for each trajectory, with all the relevant XTCs
inside that directory (PROJECT/TRAJ*/frame*.xtc).

Output: A 'Trajectories' directory containing all the merged lh5 files, and a
project info file containing information MSMBuilder uses in subsequent
calculations.\n"""

    arglist=["projectfn", "PDBfn", "input", "source", "discard", "mingen", "stride", "rmsdcutoff" ]
    options=ArgLib.parse(arglist,  Custom=[
        ("input", "Path to the parent directory containing subdirectories with MD (.xtc) data. See the description above for the appropriate formatting for directory architecture.", None),
        ("rmsdcutoff", "A safe-guard that discards any structure with and RMSD higher than the specified value (in nanometers, w/r/t the input PDB file). Pass -1 to disable this feature Default: -1 (off)", "-1")])
    print sys.argv

    ProjectFilename=options.projectfn
    PDBFilename=options.PDBfn

    rmsd_cutoff=float(options.rmsdcutoff)
    if rmsd_cutoff<=0.:
        rmsd_cutoff=1000.
    else:
        print "WARNING: Will discard any frame that is %f nm from the PDB conformation..." % rmsd_cutoff

    run( ProjectFilename, PDBFilename, options.input, options.source,
         int(options.discard), int(options.mingen), int(options.stride), rmsd_cutoff )
