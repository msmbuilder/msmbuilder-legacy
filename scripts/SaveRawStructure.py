import os
import mdtraj as md
import numpy as np
import pandas as pd
from msmbuilder import io
from msmbuilder import arglib
from msmbuilder import Project


parser = arglib.ArgumentParser(description='''
Yank randomly selected conformations from each state in a model by going
back to the original XTC files from which the project was built.

For example, if ConvertDataToHDF.py was used to strip the waters from the
XTC files, SaveStructures.py will output the PDBs *without* the water,
whereas this script will output PDBs *with* the original water, directly
from the XTC files.
''')
parser.add_argument('conformations_per_state', type=int, help='Number of conformations to sample from each state.', default=5)
parser.add_argument('assignments', default='Assignments.h5', help='Path to assignments file')
parser.add_argument('project')
parser.add_argument('output_dir', help='Location to save results')
parser.add_argument('states', nargs='+', type=int,
        help='''Which states to sample from. Pass a list of integers, separated
        by whitespace.''')
args = parser.parse_args()
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)


afile = io.loadh(args.assignments)
project = Project.load_from(args.project)

try:
    assignments = afile['arr_0']
except KeyError:
    assignments = afile['Data']
pdb = md.load(project.conf_filename)
xtclengths = {}

confs_to_grab = []

for state in args.states:
    rows, cols = np.where(assignments == state)
    permutation = np.random.permutation(len(rows))[:args.conformations_per_state]
    rows = rows[permutation]
    cols = cols[permutation]

    for traj, frame in zip(rows, cols):
        xtcfiles = project._traj_converted_from[traj]
        for xtcfn in xtcfiles:
            if xtcfn not in xtclengths:
                xtclengths[xtcfn] = len(md.load(xtcfn, top=pdb))

        l = np.concatenate(([0], np.cumsum([xtclengths[xtcfn] for xtcfn in xtcfiles])))
        xtc_trajid = np.max(np.where(frame >= l)[0])    
        xtc_frameid = frame - l[xtc_trajid]

        confs_to_grab.append((state, xtcfiles[xtc_trajid], xtc_frameid))




df = pd.DataFrame(confs_to_grab, columns=['state', 'trjfn', 'index'])
gby = df.groupby('trjfn')
counter = {s: 0 for s in args.states}

for trjfn in gby.groups.keys():
    traj = md.load(trjfn, top=pdb)
    for i, row in gby.get_group(trjfn).iterrows():
        state = row['state']
        index = row['index']
        frame = traj[index]
        out_filename = 'State%d-%d.pdb' % (state, counter[state])
        frame.save(os.path.join(args.output_dir, out_filename))

        print 'Saving frame=%d from trj=%s in state=%d as %s' % \
            (index, trjfn, state, out_filename)
        counter[state] += 1




