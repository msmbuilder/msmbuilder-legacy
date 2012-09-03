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
from argparse import ArgumentParser

from msmbuilder import License, Citation, metrics
from schwancrtools import metrics_Drift, metrics_PCA, metrics_Dihedral, additional_metrics
import numpy as np

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

'absmin': ('--ab','--abs-min','Absolute minimum to use in picking eigenvectors. Will keep all eigenvectors with eigenvalues >= --abs-min',None),

'assignments': ('-a', '--assignments', 'Path to assignments file. Default: Data/Assignments.h5', 'Data/Assignments.h5'),

'angles': ('-n','--angles','Angles to use in calculating dihedrals. Can be one or many of (phi,psi,omega,chi) separated by whitespace',None,'+'),

'clustcutoff': ('--c-cut','--cluster-cutoff','Generic cutoff to pass to a clustering script. NOTE: This is usually a number in (0,1] for some metrics. Be sure you know what the metric requires. For example, Cluster_PCA.py multiplies this number by the total variance in the dataset to pass to the K-Centers clustering',None),

'cm_scheme': ('--cm','--cm-scheme','Scheme to use in calculating a contact mape. Should be one of: CA, closest, closest-heavy','closest-heavy'),

'explvar': ('--ev','--exlp-var',"Explained variance in (0,1] to decide how many PCs to use", None ),

'generators': ('-g', '--generators', 'Path to generators file. Default: Data/Gens.lh5', 'Data/Gens.lh5'),

'gen_eps': ('--ge','--gen-eps','Path to generators\' epsilon values (ONLY FOR DRIFT METRIC). This serves as the input and output flag',None),

'atomindices': ('-i', '--indices', 'Atom indices file. Default: "AtomIndices.dat"', 'AtomIndices.dat'),

'clusters': ('-k', '--clusters', 'Maximum number of clusters (microstates) in MSM. Default: 10000000.', '10000000'),

'lagtime': ('-l', '--lagtime', "Lag time to use in model (in number of snapshots. EG, if you have snapshots every 200ps, and set the lagtime=50, you'll get a model with a lagtime of 10ns).", None),

'numvecs': ('--nv', '--num-vecs', "Number of eigenvectors to use from the PCA decomposition", None),

'output': ('-o', '--output', 'Name of file to write output to, a flat text file containing the data in NumPy savetxt/loadtxt format (.dat).', 'NoOutputSet'),

'projectfn': ('-p', '--projectfn', 'Project filename. Should have extension .h5. Default: "ProjectInfo.h5"', 'ProjectInfo.h5'),

'procs': ('-P', '--procs', 'Number of physical processors/nodes to use. Default: 1', '1'),

'pcaobject': ('--pca','--pca-object','PCA object created by mdp-toolkits.',None),

'resindices': ('--ri','--resindices', 'Residue indices to use in calculating contact maps. Should be a list of pairs of residues (ZERO-INDEXED).',None),

'reciprocal': ('--rec','--reciprocal', 'Use the reciprocal of all distances as opposed to the actual distance. This essentially makes the difference in large distances less meaningful.', False),

'PDBfn': ('-s', '--pdb', 'PDB structure file', None),

'stride': ('-u', '--subsample', 'Integer number to subsample by. Every "u-th" frame will be taken from the data and used. Default: 1', '1'),

'timecutoff': ( '--tc','--time-cutoff','Cutoff in units of frames.',None),

'tmat': ('-T', '--matrix', 'Transition matrix to build a model from, in scipy sparse format (.mtx). Default: Data/tProb.mtx', 'Data/tProb.mtx'),

'trj_dir': ('-t','--traj-dir','Directory to find the trajectories in lh5 format','./Trajectories'),

'outdir': ('-w', '--outdir', 'Location to save results.  Default: "./Data/"', './Data/'),

}

MetricReqs = {

'rmsd': (['atomindices'], 'RMSD Between atom positions in the protein conformations'),

'dihedrals': (['angles'], 'Euclidean distance between the sets of cosines and sines of the dihedrals'),

'dihedral_angles': (['angles'], 'Euclidean distance between sets of dihedral ANGLES. i.e. Don\'t use this because -179 and 179 are far...but should be close.'),

'boolean_contact': (['resindices','cm_scheme'], 'L1 norm on the difference in two contact maps'),

'continuous_contact': (['resindices', 'cm_scheme','reciprocal'], 'L2 norm on the difference in two intra-residue distance matrices')

}

DoubleMetricReqs = {

'drift': (['timecutoff','gen_eps'],'Normalize another metric by the "velocity" in that region of space'),

'state_drift':(['timecutoff'],'Normalize another metric by the diffusion constant in that state'),

'pca': (['pcaobject','PDBfn','atomindices','absmin','explvar','numvecs'],'Use PCA to calculate the distance between conformations.')

}

def parse(arglist, new_arglist=[], Custom=None, metric_parsers=False):
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
    ArgsParsed = []
    # initialize parser 
    parser = ArgumentParser()

    
    if metric_parsers:
        subparsers = parser.add_subparsers(title='Available Metrics to Use',dest='metric')
        add_args( subparsers, MetricReqs ) # Add the regular metrics
        add_args( subparsers, DoubleMetricReqs ) # Add the refinement metrics
        for sup_metric in DoubleMetricReqs.keys(): # Need to add regular metrics to each refinement metric
            sub_subparsers = subparsers.choices[sup_metric].add_subparsers(title='Available Metrics to Refine Using the %s Metric' % sup_metric, dest='base_metric') 
            # Create a subparser for each of the refinement metriccs
            add_args( sub_subparsers, MetricReqs ) # Can refine the regular ones
            add_args( sub_subparsers, DoubleMetricReqs ) # can refine the refinement too!

            sub_sub_subparsers = sub_subparsers.choices[sup_metric].add_subparsers(title='Available Metrics to Refine Using the %s Metric' % sup_metric, dest='sub_base_metric') 
            # Add subparser to each refinement subparser
            add_args( sub_sub_subparsers, MetricReqs ) # Only doing one layer. I don't think it make sense to do more than this 
            # So you can refine a refinment of a regular metric. But cannot refine a refinement of another refinement

    # Set Custom specified argument options
    if Custom:
        for CustomArg in Custom:
            CustomArgName = CustomArg[0]
            if CustomArgName in A.keys():
                A[CustomArgName] = (A[CustomArgName][0], A[CustomArgName][1], CustomArg[1], CustomArg[2])
    # Parse Arguments
    for arg in arglist:
        parser.add_argument(A[arg][0], A[arg][1], dest=arg, help=A[arg][2], default=A[arg][3])
 
    for new_arg in new_arglist:
        parser.add_argument( new_arg[1], new_arg[2], dest=new_arg[0],help=new_arg[3],default=new_arg[4])

    options = parser.parse_args()
    # Raise errors if there is something missing
    for arg in arglist:
        if not eval('options.'+arg):
            if A[arg][3] == None:
                print "ERROR: Missing required argument:", arg
                sys.exit(1)

    if metric_parsers:
        metric = initialize_metric( options.metric, options )
        return options, metric
        
    else:
        return options


def add_args( subparser, metric_dict ):

    for metric in metric_dict.keys():
        parser_metric = subparser.add_parser(metric, help=metric_dict[metric][1])
        for arg in metric_dict[metric][0]:
            if len(A[arg])==5:
                parser_metric.add_argument(A[arg][0], A[arg][1], dest=arg, help=A[arg][2], default=A[arg][3], nargs=A[arg][4] )
            elif str(A[arg][3]) == 'True':
                parser_metric.add_argument(A[arg][0], A[arg][1], dest=arg, help=A[arg][2], default=A[arg][3], action='store_false')
            elif str(A[arg][3]) == 'False':
                parser_metric.add_argument(A[arg][0], A[arg][1], dest=arg, help=A[arg][2], default=A[arg][3], action='store_true')
            else:
                parser_metric.add_argument(A[arg][0], A[arg][1], dest=arg, help=A[arg][2], default=A[arg][3] )


def initialize_metric( metric_str, options ):
    if metric_str.lower() == 'rmsd':
        try: aind=np.loadtxt(options.atomindices).astype(int)
        except: aind=None
        return metrics.RMSD( atomindices = aind )
    elif metric_str.lower() == 'dihedrals':
        return metrics.Dihedral( angles = options.angles )
    elif metric_str.lower() == 'dihedral_angles':
        return metrics_Dihedral.DihedralAngle( angles = options.angles )
    elif metric_str.lower() == 'continuous_contact':
        try: contacts = np.loadtxt( options.resindices ).astype(int)
        except: contacts = 'all'
        if options.reciprocal: # use reciprocal distances instead of regular ones
            return additional_metrics.RecContinuousContact( contacts = contacts, scheme=options.cm_scheme )
        return metrics.ContinuousContact( contacts = contacts, scheme=options.cm_scheme )
    elif metric_str.lower() == 'boolean_contact':
        try: contacts = np.loadtxt( options.resindices ).astype(int)
        except: contacts = 'all'
        return metrics.BooleanContact( contacts = contacts, scheme=options.cm_scheme )
    elif metric_str.lower() == 'drift':
        return metrics_Drift.DriftMetric( initialize_metric( options.base_metric, options ), options.timecutoff )
    elif metric_str.lower() == 'state_drift':
        return state_dependent.StateScaled( initialize_metric( options.base_metric, options ), [ - int( options.timecutoff ), int( options.timecutoff ) ] )
    elif metric_str.lower() == 'pca':
        if options.base_metric == 'rmsd':
            return metrics_PCA.RedDimPNorm( options.pcaobject, pdbFN = options.PDBfn, num_vecs = options.numvecs, expl_var = options.explvar, abs_min = options.absmin )
        else:
            if options.base_metric.lower() == 'pca' and options.sub_base_metric:
                return metrics_PCA.RedDimPNorm( options.pcaobject, prep_with = initialize_metric( options.sub_base_metric, options ), num_vecs = options.numvecs, expl_var = options.explvar, abs_min = options.absmin )
            elif options.base_metric.lower() == 'drift':
                print "It does not make sense to refine the drift metric with PCA... Remove this layer and try again."
                exit()
            else:
                return metrics_PCA.RedDimPNorm( options.pcaobject, prep_with = initialize_metric( options.base_metric, options ), num_vecs = options.numvecs, expl_var = options.explvar, abs_min = options.absmin )
    else:
        print "Got \"%s\". Must be one of: rmsd, dihedrals, continuous_contact, boolean_contact" % metric_str.lower()

