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

import glob
import os
import sys

from msmbuilder import Serializer
from msmbuilder import Project

def run():

  # Do a little clean-up
  print "Removing all PBS STDOUT, STDERR, and log files"
  os.system("rm AssignPart*.log")
  os.system("rm AssignOnPBS*.e*")
  os.system("rm AssignOnPBS*.o*")

  # Merge the stuff!
  AssFilenameList=glob.glob("Assignments.*.Ass")
  NumPartialFiles=len(AssFilenameList)

  AssFilenameList=["Assignments.%d.Ass"%i for i in range(NumPartialFiles)]
  RMSDFilenameList=["Assignments.%d.RMSD"%i for i in range(NumPartialFiles)]
  WhichTrajsFilenameList=["Assignments.%d.WhichTrajs"%i for i in range(NumPartialFiles)]

  AssList=[]
  for Filename in AssFilenameList:
    AssList.append(Serializer.LoadData(Filename))
    print "Appended %s to final output" % Filename

  RMSDList=[]
  for Filename in RMSDFilenameList:
    RMSDList.append(Serializer.LoadData(Filename))
    print "Appended %s to final output" % Filename

  WhichTrajsList=[]
  for Filename in WhichTrajsFilenameList:
    WhichTrajsList.append(Serializer.LoadData(Filename))
    print "Appended %s to final output" % Filename
  
  AllAss,AllRMSD,AllTrajs=Project.MergeMultipleAssignments(AssList,RMSDList,WhichTrajsList)
  Serializer.SaveData("Assignments.h5",AllAss)
  Serializer.SaveData("Assignments.h5.RMSD",AllRMSD)
  Serializer.SaveData("Assignments.h5.WhichTrajs",AllTrajs)
  print "Wrote: Assignments.h5, Assignments.h5.RMSD, Assignments.h5.WhichTrajs"

  print "Cleaning..."
  for Filename in AssFilenameList + RMSDFilenameList + WhichTrajsFilenameList:
      os.system("rm %s" % Filename)
  print "Done."

  return

if __name__ == "__main__":
  if len(sys.argv) != 1:
    print """
--------------------------------------------------------------------------------
Usage: run this script in the directory containing all Assignments.*.Ass,
Assignments.*.RMSD, and Assignments.*.WhichTrajs generatred by AssignOnPBS.py
--------------------------------------------------------------------------------
\n"""
    sys.exit(1)
  run()
