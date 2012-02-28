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

import os
import sys

from msmbuilder import Project
from msmbuilder import Serializer

import ArgLib

def run(project, assignments, conformations, states):

    # Check Output
    if os.path.exists("PDBs"): print "Warning: Directory 'PDBs' already exists - will overwrite any saved content in this file with the same filename"
    else: print "Creating directory: PDBs"

    project.SavePDBs( assignments, './PDBs', conformations, States=states)


if __name__ == "__main__":
    print """
Pulls 'conformations (-c)' many random structures from each state specific by a
'project (-p)' and 'assignments (-a)'. Specify which states to get
conformations using ints seperated by spaces with the states flag.\n
    EG: --states 0 1 2 3
    OR: --states all
\nOutput: A bunch of PDB files named: State<StateIndex>-<Conformation>, inside
the directory 'PDBs'
\nNote: If you want to get structures for all states, it is more efficient
to Run GetRandomConfs.py and pull structures straight from there (use the
.SaveToPDB() method and parse the output).\n"""

    arglist=["projectfn", "assignments", "conformations", "states"]
    options=ArgLib.parse(arglist, Custom=[("assignments", "Path to assignments file. Default: Data/Assignments.Fixed.h5", "Data/Assignments.Fixed.h5")])
    print "\nSPECIAL ARG: '--states', see above\n"
    print sys.argv

    for i in range(len(sys.argv)):
        if sys.argv[i] == '--states': 
             argind = i+1
        elif sys.argv[i] == '-H': 
             argind = i+1
        elif sys.argv[i] == '-a': argmax = i
        elif sys.argv[i] == '-assignments': argmax = i
        elif sys.argv[i] == '-c': argmax = i
        elif sys.argv[i] == '-conformations': argmax = i
        elif sys.argv[i] == '-p': argmax = i
        elif sys.argv[i] == '-projectfn': argmax = i
    if argmax < argind: argmax = len(sys.argv)

    if options.states in ['All', 'all', 'ALL']:
        print "Ripping PDBs for all states"
        states = None #this is the cue for the SavePDBs() method to grab all states
    else:
        print argmax
        states = [ int(state) for state in sys.argv[argind:argmax] ]
        print "Ripping PDBs for states:", states

    assignments = Serializer.LoadData(options.assignments)
    P1=Project.Project.LoadFromHDF(options.projectfn)
    run(P1, assignments, int(options.conformations), states)
