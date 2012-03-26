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

"""Library of all the arguments for MSMb2 Scripts.
"""

import sys
import os
from optparse import OptionParser, OptionGroup

from Emsmbuilder import License, Citation

# A helper function used by many msmbuilder scripts
def CheckPath(Filename):
    Dir=os.path.dirname(Filename)
    if not os.path.exists(Dir) and Dir not in [""," "]:
        print("Creating directory %s"%Dir)
        try:
            os.mkdir(Dir)
        except:
            print("Error creating directory %s; exiting!"%Dir)
            sys.exit()
    if os.path.exists(Filename):
        print("Error: %s already exists; exiting!"%Filename)
        sys.exit()

# A is a dict that holds all the argument information. The idea is that
# to edit an argument, all you have to do is change this dict.
# FORMAT: A['arg_name'] = ('-a', '--argument', 'help!', default')
# --- this list should be in alphabetical order by SHORT FLAG

A={

'assignments': ('-a', '--assignments', 'Path to assignments file. Default: Data/Assignments.h5', 'Data/Assignments.h5'),

'atomtype': ('-A', '--atoms', 'Atoms to include in index file. One of four options: (1) minimal (CA, CB, C, N, O, recommended), (2) heavy, (3) alpha (carbons), or (4) all.  Use "all" in cases where protein nomenclature may be inapproprate, although you may want to define your own indices in such situations.', 'minimal'),

'prior': ('-b', '--prior', 'Strength of Symmetric Prior. This prior mitigates the effect of sinks when estimating a reversible counts matrix (MLE Estimator). Default: 0.0', '0.0'),

"mingen": ("-B", "--mingen", "Minimum number of XTC frames required to include data in Project.  Used to discard extremely short trajectories.  Only allowed in conjunction with source = 'FAH'. Default: 0", "0"),

'conformations': ('-c', '--conformations', 'Number of conformations to randomly sample from your data', None),

'mincounts': ('-C', '--mincounts', 'Dictate that each state has a minimum number of counts. If there are states with fewer counts, merge them into other states in a greedy kinetic manner until the minimum number of counts is reached. Default: 0.', '0'),

'clustcutoff': ('--c-cut','--cluster-cutoff','Generic cutoff to pass to a clustering script. NOTE: This is usually a number in (0,1] for some metrics. Be sure you know what the metric requires. For example, Cluster_PCA.py multiplies this number by the total variance in the dataset to pass to the K-Centers clustering',None)

'discard': ('-d', '--discard', 'Number of trajectories to discard from the beginning of every XTC. Default: 0.', '0'),

'assrmsd': ('-D', '--AssignmentRMSD', 'Path to assignment RMSD file. Default: Data/Assignments.h5.RMSD', 'Data/Assignments.h5.RMSD'),

'eigvals': ('-e', '--eigenvalues', 'Number of eigenvalues (implied timescales) to retrieve at each lag time. Default: 10.', '10'),

'epsilon': ('-E', '--epsilon', 'Cutoff for merging SL states.', '0.1'),

'frequency': ('-f', '--frequency', 'Frequency with with to sample snapshots. 1=every snapshot, 2=ever other snapshot, etc... Default: 1', '1'),

'ending': ('-F', '--ending', 'Vector of states in the ending/products/folded ensemble. Default: F_states.dat.', 'F_states.dat'),

'generators': ('-g', '--generators', 'Path to generators file. Default: Data/Gens.lh5', 'Data/Gens.lh5'),

'globalkmediods': ('-G', '--globkmed', 'number of iterations of global (hybrid) kmediods to run. Default: 0.', '0'),

'states': ('-H', '--states', 'which states to sample from', None),

'atomindices': ('-i', '--indices', 'Atom indices file. Default: "AtomIndices.dat"', 'AtomIndices.dat'),

'input': ('-I', '--input', 'Input data', None),

'clusters': ('-k', '--clusters', 'Maximum number of clusters (microstates) in MSM. Default: 10000000.', '10000000'),

'checkpoint': ('-K', '--checkpoint', 'Path/name to a checkpoint file. If no file exists at that location, will write checkpoint files there. If there is a file at that location, will attempt to read that file and restart from there. Default: None (off).', "None"),

'lagtime': ('-l', '--lagtime', "Lag time to use in model (in number of snapshots. EG, if you have snapshots every 200ps, and set the lagtime=50, you'll get a model with a lagtime of 10ns).", None),

'localkmediods': ('-m', '--lockmed', 'Number of iterations of local (hybrid) k-medoids to perform. Default: 10.', '10'),

'macrostates': ('-M', '--macrostates', 'Number of macrostates in MSM', None),

'number': ('-n', '--number', 'Number of times (intervals) to calculate lagtimes for. Default: 20', '20'),

'procid': ('-N', '--procid', 'In a multi-node job, which node is currently running.', None),

'output': ('-o', '--output', 'Name of file to write output to, a flat text file containing the data in NumPy savetxt/loadtxt format (.dat).', 'NoOutputSet'),

'populations': ('-O', '--populations', 'State equilibrium populations file, in numpy .dat format. Default: Data/Populations.dat', 'Data/Populations.dat'),

'projectfn': ('-p', '--projectfn', 'Project filename. Should have extension .h5. Default: "ProjectInfo.h5"', 'ProjectInfo.h5'),

'procs': ('-P', '--procs', 'Number of physical processors/nodes to use. Default: 1', '1'),

'pcaobject': ('--pca','--pca-object','PCA object created by mdp-toolkits.',None),

'whichqueue': ('-q', '--whichqueue', 'Which PBS queue to use for jobs. Default: "long"', 'long'),

'trajlist': ('-Q', '--trajlist', 'Path to MSMBuilder1-style trajlist.', None),

'rmsdcutoff': ('-r', '--rmsdcutoff', 'Terminate k-centers clustering with rmsd cutoff when all cluster radii are below the specified size (in nanometers). This is useful for ensuring that states are small and kinetically related. Pass "-1" to force-disable this option and default to the maximum number of clusters (-k option).  Pass "0" to skip k-centers entirely, instead grabbing evenly spaced (in time) clusters.', None),

'permuteatoms': ('-R', '--permute', 'Atom labels to be permuted.', 'PermuteAtoms.dat'),

'PDBfn': ('-s', '--pdb', 'PDB structure file', None),

'source': ('-S', '--source', 'Data source: "file", "file_dcd" or "FAH". This is the style of trajectory data that gets fed into MSMBuilder. If a file, then it requires each trajectory be housed in a separate directory like (PROJECT/TRAJ*/frame*.xtc). If FAH, then standard FAH-style directory architecture is required. Default: "file"', 'file'),

'dt': ('-t', '--time', 'Time between snapshots in your data', None),

'tmat': ('-T', '--matrix', 'Transition matrix to build a model from, in scipy sparse format (.mtx). Default: Data/tProb.mtx', 'Data/tProb.mtx'),

'timecutoff': ('--t-cut','--time-cutoff','Time cutoff to use in determining the epsilon neighborhood. This is in units of frames.', None),

'stride': ('-u', '--subsample', 'Integer number to subsample by. Every "u-th" frame will be taken from the data and used. Default: 1', '1'),

'starting': ('-U', '--starting', 'Vector of states in the starting/reactants/unfolded ensemble. Default: U_states.dat', 'U_states.dat'),

'outdir': ('-w', '--outdir', 'Location to save results.  Default: "./Data/"', './Data/'),

'altindices': ('-W', '--altidx', 'Alternate atom indices for RMSD (the default set will be used for alignment).', 'AtomIndices-Alt.dat'),

'xtcframes': ('-x', '--discard', 'Number of frames to discard from the end of XTC files. MSMb2 will disregard the last x frames from each XTC. NOTE: This has changed from MSMb1. Default: 0', "0"),

'interval': ('-X', '--interval', 'The time interval spacing for calcuating lag times. IE, if -X = 5, then the script will calculate a lag time at each 5th snapshot timestep (5 snapshots, 10, 15, ...). Default: 5', '5'),

'symmetrize': ('-y', '--symmetrize', "Method by which to estimate a symmetric counts matrix. Options: 'MLE', 'MLE-TNC', 'Transpose', 'None'. Symmetrization ensures reversibility, but may skew dynamics. We recommend maximum likelihood estimation (MLE) when tractable, else try Transpose. It is strongly recommended you read the documentation surrounding this choice. Default: MLE.", 'MLE'),

'simplex': ('-Y', '--simplex', 'Specify the PCCA algorithm to use, either: "regular" or "simplex". Default: regular', 'regular'),

'datatype': ('-z', '--compressiontype', 'Format to store data in, one of: lh5, h5, xtc. Default: lh5', 'lh5'),

'directed': ('-Z', '--directed', 'Make the graph directed (if, e.g., a net flux matrix).', 'False')

}

def parse(arglist, Custom=None):
    """
    Provides for parsing within an MSMb2 script. Just call parse(['arg1', 'arg2', ...])

    Custom kwarg can be used to customize the helpstring and default for a specific arg. Pass
    arguments to Custom as a list of tuples of the following form:
    Custom= [ (ArgName (str), New Arg Description (str), New Default (any)), (...), ... ]

    eg: Custom=[("output", "Output: The data file to write", "MyData.dat")]
    """

    # Print license and citation strings, but only if parse() gets called
    print License.LicenseString
    print Citation.CiteString
   
    # initialize parser 
    parser = OptionParser(usage="%prog [options]")
    required = OptionGroup(parser, "Required Arguments")
    optional = OptionGroup(parser, "Arguments with Defaults (Defaults Represent Recommended Values)")

    # Set Custom specified argument options
    if Custom:
        for CustomArg in Custom:
            CustomArgName = CustomArg[0]
            A[CustomArgName] = (A[CustomArgName][0], A[CustomArgName][1], CustomArg[1], CustomArg[2])

    # Parse Arguments
    for arg in arglist:
        if A[arg][3] == None: 
            required.add_option(A[arg][0], A[arg][1], dest=arg, help=A[arg][2], default=A[arg][3])
        else:
            optional.add_option(A[arg][0], A[arg][1], dest=arg, help=A[arg][2], default=A[arg][3])
    parser.add_option_group(required)
    parser.add_option_group(optional)
    (options, args) = parser.parse_args()

    # Raise errors if there is something missing
    for arg in arglist:
        if not eval('options.'+arg):
            if A[arg][3] == None:
                print "ERROR: Missing required argument:", arg
                sys.exit(1)

    return options
