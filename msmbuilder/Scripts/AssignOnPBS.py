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
#

import sys
import os
import glob

import ArgLib

def run(ProjectFilename, GenFilename, IndFilename, procs,WhichQueue):
  # Check Output
  if len(glob.glob('./AssignOnPBS*.sh')) != 0:
    print "Error: There are already 'AssignOnPBS*.sh' files present! Exiting."
    sys.exit(1)
  
  # Create a bunch of AssignOnCertaintyXX.sh scripts and submit them
  for i in range(procs):
    argstring = "-p %s -g %s -i %s -P %d -N %d" % (ProjectFilename, GenFilename, IndFilename, procs, i)
    cmd='MA.py %s >& AssignPart%d.log' % (argstring, i)
    curDir = os.path.abspath(".")
    name="AssignPart%d" % i

    PBS_File="""#!/bin/bash

#PBS -N %s
#PBS -e %s/AssignOnPBS%d.sh.err
#PBS -o %s/AssignOnPBS%d.sh.out
#PBS -l nodes=%d:ppn=%d
#PBS -l walltime=%s
#PBS -V


PBS_O_WORKDIR='%s'
export PBS_O_WORKDIR
### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------

        echo The master node of this job is `hostname`
        echo The working directory is `echo $PBS_O_WORKDIR`
        echo This job runs on the following nodes:
        echo `cat $PBS_NODEFILE`


### end of information preamble
cd $PBS_O_WORKDIR

# execute commands
%s
""" % (name, curDir, i, curDir, i, 1, 1, "20:00:00", curDir, cmd)

    fn="AssignOnPBS%s.sh" % i
    f=open(fn, 'w')
    f.write(PBS_File)
    f.close()
    print "Wrote, Sumbitted, Removed:", fn
    os.system("qsub -q %s %s" % (WhichQueue,fn))
    os.system("rm %s" % fn)

  return


if __name__ == "__main__":
  print """\nAssigns any data not used in original clustering to generators. Does
this in parallel on a PBS cluster, dividing work amongst many nodes (specify
with -P option). NOTE: The maximum available parallelization is one machine per
trajectory. After this script's children on the nodes completes, run
MergeAssign.py, and the final output is:
-- Assignments.h5: a matrix of assignments where each row is a vector
corresponding to a data trajectory. The values of this vector are the cluster
assignments.
-- Assignments.h5.RMSD: Gives the RMSD from the assigned frame to its Generator.
-- Assignments.h5.WhichTrajs: Shows which trajectories were used (if
subsampling, disabled in this wrapper script).\n"""

  arglist=["projectfn", "generators", "atomindices", "procs","whichqueue"]
  options=ArgLib.parse(arglist)
  print sys.argv
  
  run(options.projectfn, options.generators, options.atomindices, int(options.procs), options.whichqueue)
