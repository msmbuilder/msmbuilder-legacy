"""
Robert McGibbon
Pande Lab
Stanford University
July 2011

Wrapper to convert the XTC files from a FAH project to the LH5 format
for msmbuilder2.

Small Edits: TJL <tjlane@stanford.edu>
"""

# GLOBAL IMPORTS
import os
import re
import sys
from numpy import argmax

import multiprocessing
from multiprocessing import Pool

from msmbuilder import Trajectory
from msmbuilder import DistanceMetric
from msmbuilder import Conformation
from msmbuilder import Project

import ArgLib


def run(input_dir, output_dir, pdb_file, xtc_filename_root, output_filename_root,
        output_filename_type, stride, max_rmsd, discard_first_N_snapshots, max_gens,
        min_gens, center_conformations, num_proc, project_filename):
    """Convert all of the trajectories in the FAH project in input_dir to
    lh5 trajectory files which will be placed in output dir.

    The pdb file is necessary for atom definitions and such.

    Uses num_proc processors, which have to all be on the same node,
    as this wrapper uses multiprocessing.Pool

    For a description of the rest of the input parameters, just run the code
    with the -h flag.
    """

    try:
        os.mkdir(output_dir)
    except OSError:
        print >> sys.stderr, 'ERROR: The directory %s already exists' % output_dir
        sys.exit(1)

    # Because some RUN/CLONE folders will not yield a trajectory for various reasons
    # the trajectory files numbering will have some gaps in it, which is not good for
    # various other msmbuilder scripts. So we dont want to name the files created in the
    # next step with their final name.
    intermediate_filename_root = '_%s' % output_filename_root

    jobs = []
    for i, clone_dir in enumerate(yield_clone_directories(input_dir)):
        job = {'clone_dir' : clone_dir,
               'output_dir': output_dir,
               'pdb_file': pdb_file,
               'trajectory_number': i,
               'xtc_filename_root': xtc_filename_root,
               'output_filename_root': intermediate_filename_root,
               'output_filename_type': output_filename_type,
               'stride': stride,
               'max_rmsd': max_rmsd,
               'discard_first_N_snapshots': discard_first_N_snapshots,
               'max_gens': max_gens,
               'min_gens': min_gens,
               'center_conformations': center_conformations
               }
        jobs.append(job)
            
    if num_proc > 1:
        # use multiprocessing
        pool = Pool(processes=num_proc)
        pool.map(create_trajectory, jobs)
    else:
        # use regular serial execution
        map(create_trajectory, jobs)


    # Because some RUN/CLONE folders did not yield a trajectory for various reasons
    # the trajectory files numbering may have some gaps in it, which is not good for
    # various other msmbuilder scripts. So now we have to rename them to be sequential
    print "Finished Generating Trajectories. Renaiming them now into contiguous order"
    for i, filename in enumerate(os.listdir(output_dir)):
        path = os.path.join(output_dir, filename)
        new_path = os.path.join(output_dir, "%s%d%s" % (output_filename_root, i, output_filename_type))
        os.rename(path, new_path)

    print "Generating Project File..."
    Project.CreateProjectFromDir(Filename=project_filename,
                                 TrajFilePath=output_dir,
                                 TrajFileBaseName=output_filename_root,
                                 TrajFileType=output_filename_type,
                                 ConfFilename=pdb_file)

    print "All finished! Your next step is probably to create atom indicies (CreateAtomIndicies.py)"
    

def create_trajectory(args):
    '''
    This function takes in a path to a CLONE and merges all the XTC files
    it finds into a LH5 trajectory

    Because it is designed to be called via map(),
    which only supports supplying 1 iterable to the function, this function
    must formally only take 1 argument.

    The necessary and required items in the args hash are as follows:

    clone_dir: (string) the directory in which the xtc files are found. All
       of the xtc files in this directory are joined together to make
       a single trajectory (.lh5) output file
    output_dir: (string) directory where the outputted files will be placed
    pdb_file: (string) path to a pdb file with the correct atom labels and such
    trajectory_number: A unique number for this trajectory. This number
       is used in constructing the filename to write the outputted .lh5
       trajectory to, and thus must be unique
    xtc_filename_root:  (string) The stem of the xtc files filenames
    output_filename_root: (string) The stem of the outputted trajectory
       files
    output_filename_type: (string) The extentions of the outputted trajectories
    stride: (integer) Subsample by only considering every Nth snapshop.
    max_rmsd: (integer/None) if this value is not None, calculate the RMSD to
       the pdb_file from each snapshot and reject trajectories which have snapshots with RMSD
       greated than max_rmsd. If None, no check is performed
    discard_first_N_snapshots: (integer) ignore the first N snapshots of each trajectory
    max_gens: (integer/None) if this value is not None, discard data after
      the Nth XTC file. If it is none, no check is performed
    min_gens: (integer) Discard the trajectories that contain fewer than N XTC files.
    center_conformations: (boolean) center conformations before saving.
    '''
    try:
        clone_dir = args['clone_dir']
        output_dir = args['output_dir']
        pdb_file = args['pdb_file']
        trajectory_number = args['trajectory_number']
        xtc_filename_root = args['xtc_filename_root']
        output_filename_root = args['output_filename_root']
        output_filename_type = args['output_filename_type']
        stride = args['stride']
        max_rmsd = args['max_rmsd']
        discard_first_N_snapshots = args['discard_first_N_snapshots']
        max_gens = args['max_gens']
        min_gens = args['min_gens']
        center_conformations = args['center_conformations']
    except KeyError as e:
        print >> sys.stderr, """One or more required keys (%s) was not
        suplied in the input argument to create_trajectory().""" % e
        sys.exit(1)

    # only process files named <xtc_filename_root>XX.xtc
    # where XX is some number
    pattern = re.compile('%s(\d+)[.]xtc' % xtc_filename_root)

    xtc_files = [e for e in os.listdir(clone_dir) if pattern.search(e)]

    # sort the XTC files by the number, not alphabetically
    def integer_component(filename):
        '''extract a the numeric part of the filename for sorting'''
        try:
            substr = pattern.search(filename).group(1)
            return int(substr)
        except:
            print >> sys.stderr, """A filename (%s) may not have been
            sorted correctly""" % filename
            return 0

    xtc_files.sort(key=integer_component)

    # Ensure that we're only joining contiguously numbered xtc files -- starting at 0 --
    # into a trajectory. If there are gaps in the xtc files in the directory, we only
    # want to use the the ones such that they are contiguously numbered
    i = 0
    for i, filename in enumerate(xtc_files):
        if integer_component(filename) != i:
            break
    xtc_files = xtc_files[0:i+1]
    
    if max_gens is not None:
        num_to_keep = min(len(xtc_files), max_gens)
        xtc_files = xtc_files[0:num_to_keep]

    xtc_file_paths = [os.path.join(clone_dir, f) for f in xtc_files]
    
    print "Processing %d xtc files in clone_dir = %s" % (len(xtc_files), clone_dir)

    if len(xtc_files) <= min_gens:
        print "Skipping trajectory in clone_dir = %s" % clone_dir
        print "Too few xtc files."
        return

    #swap stdout
    sys.stdout.flush()
    real_stdout = sys.stdout
    sys.stdout = open('/dev/null', 'w')
    try:
        trajectory = Trajectory.Trajectory.LoadFromXTC(xtc_file_paths, PDBFilename=pdb_file)
    except IOError as e:
        print >> sys.stderr, "IOError (%s) when processing trajectory in clone_dir = %s" % (e, clone_dir)
        print >> sys.stderr, "Attempting rescue by disregarding final frame, which is often"
        print >> sys.stderr, "the first/only frame to be corrupted"

        try:
            trajectory = Trajectory.Trajectory.LoadFromXTC(xtc_file_paths[0:-1], PDBFilename=pdb_file)
        except IOError:
            print >> sys.stderr, "Unfortunately, the error remained even after ignoring the final"
            print >> sys.stderr, "frame."
            print >> sys.stderr, "Skipping the trajectory in clone_dir = %s" % clone_dir
            return
        print >> sys.stderr, "Sucessfully recovered from IOError by disregarding final frame."

    #bring back stdout
    sys.stdout.flush()
    sys.stdout = real_stdout

    if max_rmsd is not None:
        pdb_conformation = Conformation.Conformation.LoadFromPDB(pdb_file)
        rmsds = trajectory.CalcRMSD(pdb_conformation)
        if max(rmsds) > max_rmsd:
            print >> sys.stderr, "Snapshot %d has an RMSD %f > the %f cutoff to the pdb file." % (argmax(rmsds), max(rmsds), max_rmsd)
            print >> sys.stderr, "Dropping trajectory"
            return

    if center_conformations:
        DistanceMetric.centerConformations(trajectory["XYZList"])

    output_filename = '%s%s%s' % (output_filename_root, trajectory_number, output_filename_type)
    output_file_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_file_path):
        print >> sys.stderr, "The file name %s already exists. Skipping" % output_file_path
        return

    # stide and discard by snapshot
    trajectory['XYZList'] = trajectory['XYZList'][discard_first_N_snapshots::stride]
    
    trajectory.Save(output_file_path)

    return

def yield_clone_directories(input_dir):
    """
    This function is an enumerator that yields all the CLONE directories
    inside of the FAH project whose root is input_dir.
    """
    
    input_dir_contents = os.listdir(input_dir)
    for input_dir_entry in input_dir_contents:
        run_directory = os.path.join(input_dir, input_dir_entry)
        if input_dir_entry.startswith("RUN") and os.path.isdir(run_directory):

            #look through each file in the RUN* directory
            run_directory_contents = os.listdir(run_directory)
            for run_directory_entry in run_directory_contents:
                path = os.path.join(run_directory, run_directory_entry)
                if run_directory_entry.startswith("CLONE") and os.path.isdir(path):
                    yield path
        elif input_dir_entry.startswith("CLONE") and os.path.isdir(run_directory):
            yield os.path.join(input_dir, input_dir_entry)


def boolean(string):
    '''Convert string to boolean. Used for argparse type='''
    string = string.lower()
    if string in ['0', 'f', 'false', 'no', 'off']:
        return False
    elif string in ['1', 't', 'true', 'yes', 'on']:
        return True
    else:
        raise ValueError()

      
if __name__ == '__main__':
    print """
\nMerges individual XTC files into continuous lossy HDF5 (.lh5) trajectories. Can
take either data from a FAH project (PROJECT/RUN*/CLONE*/frame*.xtc) or from a
directory containing one directory for each trajectory, with all the relevant XTCs
inside that directory (PROJECT/TRAJ*/frame*.xtc).

Output: A 'Trajectories' directory containing all the merged lh5 files, and a
project info file containing information MSMBuilder uses in subsequent
calculations.\n"""

    arglist=["projectfn", "PDBfn", "input", "source", "discard", "mingen", "stride", "assrmsd", "procs"]
    options=ArgLib.parse(arglist,  Custom=[("input", "Path to the parent directory containing subdirectories with MD (.xtc) data. See the description above for the appropriate formatting for directory architecture.", None), ("assrmsd", "RMSD cutoff. Discards any trajectory with a higher RMSD (with respect to the provided PDB) cutoff than this value. Default: 10 (nm)", "10")])
    print sys.argv

    ProjectFilename=options.projectfn
    PDBFilename=options.PDBfn

    # A few of the options below have been disabled/hardwired. They can be
    # revealed manually below. -- TJL
    run(input_dir = options.input,
        output_dir = "./Trajectories",
        pdb_file = options.PDBfn,
        xtc_filename_root = "frame",
        output_filename_root = "lh5",
        output_filename_type = "lh5",
        stride = options.stride,
        max_rmsd = options.assrmsd,
        discard_first_N_snapshots = options.discard,
        max_gens = None,
        min_gens = None,
        center_conformations = False,
        num_proc = options.procs,
        project_filename = option.projectfn)
