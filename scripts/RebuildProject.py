#!/usr/bin/env python
#"""RebuildProject.py: #
#
#"""
#import argparse
import os

from msmbuilder import Project
from msmbuilder import CreateMergedTrajectoriesFromFAH
from msmbuilder import arglib


parser = arglib.ArgumentParser(description="Search for local trajectories and create a ProjectInfo.h5 file.")
parser.add_argument('pdb')
parser.add_argument('filetype', help='Filetype of trajectories to use.', default='.lh5')
parser.add_argument('project', help='Filename of Project to output', default='ProjectInfo.h5', type=str)
args = parser.parse_args()

if not os.path.exists(args.project):
    Project.CreateProjectFromDir(Filename=args.project,
                                 ConfFilename=args.pdb,
                                 TrajFileType=args.filetype)
    print 'Created %s' % args.project
else:
    print '%s already exists.' % args.project
