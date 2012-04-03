#!/usr/bin/env python
#"""RebuildProject.py: #
#
#"""
#import argparse
import os

from Emsmbuilder import Project,
from Emsmbuilder import CreateMergedTrajectoriesFromFAH
from Emsmbuilder import arglib


parser = arglib.ArgumentParser(description="Search for local trajectories and create a ProjectInfo.h5 file.")
parser.add_argument('pdb')
parser.add_argument('filetype', description='Filetype of trajectories to use.', default='.lh5')
parser.add_argument('project', description='Filename of Project to putput', default='ProjectInfo.h5', type=str)
args = parser.parse_args()

if not os.path.exists(args.project):
    Project.CreateProjectFromDir(Filename=args.project,
                                 ConfFilename=args.pdb,
                                 TrajFileType=args.filetype)
    print 'Created %s' % args.project
else:
    print '%s already exists.' % args.project
