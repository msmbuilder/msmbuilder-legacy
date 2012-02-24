"""
Robert McGibbon
Pande Lab
Stanford University
July 2011

Wrapper to convert the XTC files from a FAH project to the LH5 format
for msmbuilder2
"""

# GLOBAL IMPORTS
import os
import re
import sys
from argparse import ArgumentParser
from numpy import argmax

import multiprocessing
from multiprocessing import Pool

from msmbuilder import Trajectory
from msmbuilder import DistanceMetric
from msmbuilder import Conformation
from msmbuilder import Project


def run(input_dir, output_dir, pdb_file, stride, max_rmsd, discard_first_N_snapshots,
        min_gens, center_conformations, num_proc, project_filename, input_style):
    """Convert all of the trajectories in the FAH project in input_dir to
    lh5 trajectory files which will be placed in output dir.

    Note: Since sometimes a conversion fails, we collect all trajectories at the
    end and renumber them such that they are contiguously numbered.
    """

    try:
        os.mkdir(output_dir)
    except OSError:
        print >> sys.stderr, 'ERROR: The directory %s already exists' % output_dir
        sys.exit(1)

    intermediate_filename_root = '_trj' # A placeholder name
    jobs = []
    for i, clone_dir in enumerate(yield_xtc_directories(input_dir, input_style)):
        #print clone_dir
        job = {'clone_dir' : clone_dir,
               'output_dir': output_dir,
               'pdb_file': pdb_file,
               'trajectory_number': i,
               'stride': stride,
               'max_rmsd': max_rmsd,
               'discard_first_N_snapshots': discard_first_N_snapshots,
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

    # Rename trajectory files such that they have contiguous numbering
    print "Finished Generating Trajectories. Renaming them now in contiguous order"
    for i, filename in enumerate(os.listdir(output_dir)):
        path = os.path.join(output_dir, filename)
        new_path = os.path.join(output_dir, "trj%d.lh5" % i)
        os.rename(path, new_path)

    print "Generating Project File..."
    Project.CreateProjectFromDir(Filename=project_filename,
                                 TrajFilePath=output_dir,
                                 TrajFileBaseName='trj',
                                 TrajFileType='.lh5',
                                 ConfFilename=pdb_file)

    print "Data converted properly."
    return
    

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
    stride: (integer) Subsample by only considering every Nth snapshop.
    max_rmsd: (integer/None) if this value is not None, calculate the RMSD to
       the pdb_file from each snapshot and reject trajectories which have snapshots with RMSD
       greated than max_rmsd. If None, no check is performed
    discard_first_N_snapshots: (integer) ignore the first N snapshots of each trajectory
    min_gens: (integer) Discard the trajectories that contain fewer than N XTC files.
    center_conformations: (boolean) center conformations before saving.
    '''
    try:
        clone_dir = args['clone_dir']
        output_dir = args['output_dir']
        pdb_file = args['pdb_file']
        trajectory_number = args['trajectory_number']
        stride = args['stride']
        max_rmsd = args['max_rmsd']
        discard_first_N_snapshots = args['discard_first_N_snapshots']
        min_gens = args['min_gens']
        center_conformations = args['center_conformations']
    except KeyError as e:
        print >> sys.stderr, """One or more required keys (%s) was not
        suplied in the input argument to create_trajectory().""" % e
        sys.exit(1)

    # only process files named <string>XX.xtc
    # where XX is some number
    pattern = re.compile('\w+(\d+)[.]xtc')
    xtc_files = [e for e in os.listdir(clone_dir) if pattern.search(e)]

    # sort the XTC files by the number, not alphabetically
    def integer_component(filename):
        '''extract a the numeric part of the filename for sorting'''
        try:
            substr = pattern.search(filename).group(1)
            return int(substr)
        except:
            print >> sys.stderr, """A filename (%s) may not have been sorted correctly""" % filename
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

    xtc_file_paths = [os.path.join(clone_dir, f) for f in xtc_files]
    
    print "Processing %d xtc files in clone_dir = %s" % (len(xtc_files), clone_dir)

    if len(xtc_files) <= min_gens:
        print "Skipping trajectory in clone_dir = %s" % clone_dir
        print "Too few xtc files (generations)."
        return

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

    if max_rmsd is not None:
        pdb_conformation = Conformation.Conformation.LoadFromPDB(pdb_file)
        rmsds = trajectory.CalcRMSD(pdb_conformation)
        if max(rmsds) > max_rmsd:
            print >> sys.stderr, "Snapshot %d has an RMSD %f > the %f cutoff to the pdb file." % (argmax(rmsds), max(rmsds), max_rmsd)
            print >> sys.stderr, "Dropping trajectory"
            return

    if center_conformations:
        DistanceMetric.centerConformations(trajectory["XYZList"])

    output_filename = 'trj%s.lh5' % trajectory_number
    output_file_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_file_path):
        print >> sys.stderr, "The file name %s already exists. Skipping it." % output_file_path
        return

    # stide and discard by snapshot
    trajectory['XYZList'] = trajectory['XYZList'][discard_first_N_snapshots::stride]
    trajectory.Save(output_file_path)

    return

def yield_xtc_directories(input_dir, input_style):
    """
    This function is an enumerator that yields all of the
    directories under the input_dir that contain XTCs to be
    merged into trajectories.

    If you use input_style = 'FAH', this code uses knowledge of the
    RUN*/CLONE* directory structure to yield all the CLONE directories.

    If you use input_style = 'FILE', this code uses os.walk() which is
    A LOT slower because it has to stat every file, but is capable of recursively
    searching for xtc files to arbitrary depths.
    """
    if input_style == 'FAH':
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

    # FAH style is faster, but FILE style is more flexable
    elif input_style == 'FILE':
        for (dirpath, dirnames, filenames) in os.walk(input_dir):
            # if any of the filenames in this directory end in .xtc, then yield this directory
            def contains_xtcs(filenames):
                for filename in filenames:
                    if filename.endswith('.xtc'):
                        return True
                return False
            
            if contains_xtcs(filenames):
                yield dirpath

    else:
        raise Exception("This function was invoked incorrectly. The only acceptable choices for 'input_style' are 'FILE and 'FAH'. You supplied %s" % input_style)

    return

      
if __name__ == '__main__':
    print """
\nMerges individual XTC files into continuous lossy HDF5 (.lh5) trajectories. Can
take either data from a FAH project (PROJECT/RUN*/CLONE*/frame*.xtc) or from a
directory containing one directory for each trajectory, with all the relevant XTCs
inside that directory (PROJECT/TRAJ*/frame*.xtc).

Output: A 'Trajectories' directory containing all the merged lh5 files, and a
project info file containing information MSMBuilder uses in subsequent
calculations.\n"""

    arglist=["projectfn", "PDBfn", "input", "source", "discard", "mingen", "stride", "rmsdcutoff", "procs"]
    options=ArgLib.parse(arglist,  Custom=[
        ("input", "Path to the parent directory containing subdirectories with MD (.xtc) data. See the description above for the appropriate formatting for directory architecture.", None),
        ("rmsdcutoff", "A safe-guard that discards any structure with and RMSD higher than the specified value (in nanometers, w/r/t the input PDB file). Default: 100.0 nm", "100.0") ])
    print sys.argv

    # Some options have been hardwired, but can be changed or revealed here  --TJL
    run(input_dir                 = options.input,
        output_dir                = "./Trajectories",
        pdb_file                  = options.PDBfn,
        stride                    = int(options.stride),
        max_rmsd                  = float(options.rmsdcutoff),
        discard_first_N_snapshots = int(options.discard),
        min_gens                  = int(options.mingen),
        center_conformations      = True,
        num_proc                  = int(options.procs),
        project_filename          = options.projectfn,
        input_style               = options.input )
    
