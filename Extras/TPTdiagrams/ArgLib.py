# Class containing a library of all the arguments for MSMb2Wrap
# Each of the MSMb2Wrap scripts gets its arguments from this class
# TJL 2011, PANDE LAB

import sys
from optparse import OptionParser, OptionGroup

CiteString="The Pande Group rocks. You should just cite all our papers as much as possible!\n"

# A is a dict that holds all the argument information. The idea is that
# to edit an argument, all you have to do is change this dict.
# FORMAT: A['arg_name'] = ('-a', '--argument', 'help!', default')

A={

'assignments': ('-a', '--assignments', 'Path to assignments file. Default: Data/Assignments.h5', 'Data/Assignments.h5'),

'assrmsd': ('-D', '--AssignmentRMSD', 'Path to assignment RMSD file. Default: Data/Assignments.h5.RMSD', 'Data/Assignments.h5.RMSD'),

'atomtype': ('-A', '--atoms', 'Atoms to include in index file. One of four options: (1) minimal (CA, CB, C, N, O, recommended), (2) heavy, (3) alpha (carbons), or (4) all.  Use "all" in cases where protein nomenclature may be inapproprate, although you may want to define your own indices in such situations.', 'minimal'),

'conformations': ('-c', '--conformations', 'Number of conformations to randomly sample from your data', None),

'discard': ('-d', '--discard', 'Number of trajectories to discard from the beginning of every XTC. Default: 0.', '0'),

'eigvals': ('-e', '--eigenvalues', 'Number of eigenvalues (implied timescales) to retrieve at each lag time. Default: 10.', '10'),

'epsilon': ('-E', '--epsilon', 'Population cutoff for including states. Default: 5%', '0.05'),

'frequency': ('-f', '--frequency', 'Frequency with with to sample snapshots. 1=every snapshot, 2=ever other snapshot, etc... Default: 1', '1'),

'starting': ('-F', '--starting', 'Vector of states in the starting/folded ensemble. Default: F_states.dat.', 'F_states.dat'),

'generators': ('-g', '--generators', 'Path to generators file. Default: Data/Gens.lh5', 'Data/Gens.lh5'),

'globalkmediods': ('-G', '--globkmed', 'number of iterations of global (hybrid) kmediods to run. Default: 0.', '0'),

'states': ('-H', '--states', 'which states to sample from', None),

'atomindices': ('-i', '--indices', 'Atom indices file. Default: "AtomIndices.dat"', 'AtomIndices.dat'),

'input': ('-I', '--input', 'Input data', None),

'interval': ('-X', '--interval', 'The time interval spacing for calcuating lag times. IE, if -X = 5, then the script will calculate a lag time at each 5th snapshot timestep (5 snapshots, 10, 15, ...). Default: 5', '5'),

'clusters': ('-k', '--clusters', 'Maximum number of clusters (microstates) in MSM. Default: 10000000.', '10000000'),

'lagtime': ('-l', '--lagtime', "Lag time to use in model (in number of snapshots. EG, if you have snapshots every 200ps, and set the lagtime=50, you'll get a model with a lagtime of 10ns).", None),

'localkmediods': ('-m', '--lockmed', 'Number of iterations of local (hybrid) k-medoids to perform. Default: 10.', '10'),

'macrostates': ('-M', '--macrostates', 'Number of macrostates in MSM', None),

'mincounts': ('-C', '--mincounts', 'Dictate that each state has a minimum number of counts. If there are states with fewer counts, merge them into other states in a greedy kinetic manner until the minimum number of counts is reached. Default: 3.', '3'),

'number': ('-n', '--number', 'Number of times (intervals) to calculate lagtimes for. Default: 20', '20'),

'output': ('-o', '--output', 'Name of file to write output to, a flat text file containing the data in NumPy savetxt/loadtxt format (.dat).', 'NoOutputSet'),

'populations': ('-O', '--populations', 'State equilibrium populations file, in numpy .dat format', 'Populations.dat'),

'projectfn': ('-p', '--projectfn', 'Project filename. Should have extension .h5. Default: "ProjectInfo.h5"', 'ProjectInfo.h5'),

'procs': ('-P', '--procs', 'Number of physical processors/nodes to use. Default: 1', '1'),

'procid': ('-N', '--procid', 'In a multi-node job, which node is currently running.', None),

'rmsd': ('-r', '--rmsd', 'Maximum RMSD distance from assignemnt to generator to keep (nm)', None),

'rmsdcutoff':('-D', '--rmsdcutoff', 'Terminate k-centers clustering with rmsd cutoff. If your cluster size gets below the specified value, k-centers will terminate. This is useful for ensuring that states are small and kinetically close (ie can interconvert rapidly).', None),

'randconfs': ('-R', '--randconfs', 'Random conformations from each cluster - Trajectory file type.', None),

'PDBfn': ('-s', '--pdb', 'PDB structure file', None),

'source': ('-S', '--source', 'Data source: file or FAH. This is the style of source data that gets fed into MSMBuilder. If a file, then it requires each trajectory be housed in a separate directory like (PROJECT/TRAJ*/frame*.xtc). If FAH, then standard FAH-style directory architecture is required. Default: "file"', 'file'),

'symmetrize': ('-y', '--symmetrize', "Method by which to estimate a symmetric counts matrix. Options: 'MLE', 'Transpose', 'None'. Symmetrization ensures reversibility, but may skew dynamics. We recommend maximum likelihood estimation (MLE) when tractable, else try Transpose. It is strongly recommended you read the documentation surrounding this choice. Default: MLE.", 'MLE'),

'dt': ('-t', '--time', 'Time between snapshots in your data', None),

'tmat': ('-T', '--Tmat', 'Transition matrix to build a model from, in scipy sparse format (.mtx). Default: tProb.mtx', 'tProb.mtx'),

'simplex': ('-Y', '--simplex', 'Specify the PCCA algorithm to use, either: "regular" or "simplex". Default: regular', 'regular'),

'stride': ('-u', '--subsample', 'Integer number to subsample by. Every "u-th" frame will be taken from the data and clustered. Assign remaining data with Assign.py or AssignOnPBS.py. Default: 1', '1'),

'ending': ('-U', '--ending', 'Vector of states in the final/unfolded ensemble. Default: U_states.dat', 'U_states.dat'),

'whichqueue': ('-q', '--whichqueue', 'Which PBS queue to use for jobs. Default: "long"', 'long'),

'trajlist': ('-Q', '--trajlist', 'Path to MSMBuilder1-style trajlist.', None),

'xtcframes': ('-x', '--discard', 'Number of frames to discard from the end of XTC files. MSMb2 will disregard the last x frames from each XTC. NOTE: This has changed from MSMb1. Default: 0', "0"),

"mingen": ("-X", "--mingen", "Minimum number of XTC frames required to include data in Project.  Used to discard extremely short trajectories.  Only allowed in conjunction with source = 'FAH'.", "0"),

'datatype': ('-z', '--compressiontype', 'Format to store data in, one of: lh5, h5, xtc. Default: lh5', 'lh5'),

'directed': ('-Z', '--directed', 'Choose if the graph is directed or not by passing -Z directed or -Z undirected.', 'undirected')

}

def parse(arglist, Custom=None):
    """
    Provides for parsing within an MSMb2W script. Just call parse(['arg1', 'arg2', ...])

    Custom kwarg can be used to customize the helpstring and default for a specific arg. Pass
    arguments to Custom as a list of tuples of the following form:
    Custom= [ (ArgName (str), New Arg Description (str), New Default (any)), (...), ... ]

    eg: Custom=[("output", "Output: The data file to write", "MyData.dat")]
    """
    
    print CiteString

    parser = OptionParser(usage="%prog [options]")
    required = OptionGroup(parser, "Required Arguments")
    optional = OptionGroup(parser, "Optional Arguments")

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
