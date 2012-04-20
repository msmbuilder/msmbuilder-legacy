"""
This is a class that contains all of the necessary tools to interact
with folding@home projects.

Currently doesn't work - need to do some testing!

TJ
"""

# GLOBAL IMPORTS
import os
import re
import sys
import cPickle
import base64

from numpy import argmax, concatenate

import subprocess
from subprocess import PIPE

import multiprocessing
from multiprocessing import Pool

from msmbuilder import Trajectory
from msmbuilder.metrics import RMSD
from msmbuilder import Conformation
from msmbuilder import Project
from msmbuilder import Serializer


class FahProject(object):
    """ A generic class for interacting with Folding@home projects """

    def __init__(self, pdb, project_number=0001, projectinfo_file="ProjectInfo.h5", work_server=None, email=None):

        # metadata associated with a FAH project
        self.project_number   = project_number
        self.work_server      = work_server
        self.pdb_topology     = pdb
        self.manager_email    = email
        self.projectinfo_file = projectinfo_file

        # load in the memory state
        if os.path.exists( projectinfo_file ):
            self.load_memory_state( projectinfo_file )
        else:
            self.memory = {}
            print "\nNo file: %s found. Generating new memory state." % projectinfo_file
            print "Processing all trajectories, will save progress to: %s\n" % projectinfo_file

        # set the nested classes defined below to be separate namespaces
        # this should separate concerns and prevent (dangerous) user mistakes
        self.retrieve = FahProject._retrieve(self)
        self.inject   = FahProject._inject(self)
            

    def restart_server(self):
        """
        Restarts the workserver, should be called when injecting runs.

        Checks that the server comes back up without throwing an error -
        if it doesn't come up OK, sends mail to the project manager.
        """
        raise NotImplementedError()
        print "Restarting server: %s" % self.work_server
        cmd = "/etc/init.d/FAHWorkServer-%s restart" % self.work_server
        r = subprocess.call("grompp -h", shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        return  


    def save_memory_state(self):
        """ when saving, we encode the keys into base16 with a leading 'a',
        because the HDF5 Serializer doesnt' like '/' characters and is
        super picky in general """
            
        project_info = Project.LoadFromHDF( projectinfo_file )
        project_info["Memory"] = cPickle.dumps( self.memory )
        project_info.SaveToHDF( projectinfo_file, do_file_check=False )

        return  


    def load_memory_state(self, projectinfo_file):
        """ When reading the memory state, we have to decode base16, and
        also remove leading 'a' characters """

        print "\nLoading memory state from: %s" % projectinfo_file

        project_info = Project.LoadFromHDF( projectinfo_file )
        self.memory = cPickle.loads( project_info["Memory"] )

        return


    class _retrieve():
        """ Contains all of the functions necessary for getting data from
        the workserver. Functions are able to navagate the WS directory
        structure and get trajectories and other information """


        # un-nest the local namespace from the parent class
        def __init__(self, outerclass):
            self.__dict__.update(outerclass.__dict__)
            self.save_memory_state = outerclass.save_memory_state
            
            
        def write_all_trajectories(self, input_dir, output_dir, stride, max_rmsd,
                                   discard_first_N_snapshots, min_gens,
                                   center_conformations, num_proc, 
                                   input_style, update=False):
            """
            Convert all of the trajectories in the FAH project in input_dir to
            lh5 trajectory files which will be placed in output dir.

            Note: Since sometimes a conversion fails, we collect all trajectories at the
            end and renumber them such that they are contiguously numbered.

            If the 'update' flag is set, then will use the memory object to check for
            previously converted data, and add to it (rather than reconverting everything).
            This functionality can be more cleanly called through the update_trajectories()
            method.
            """

            if update:
                assert os.path.exists(output_dir)
            else:
                try:
                    os.mkdir(output_dir)
                except OSError:
                    print >> sys.stderr, 'Error: The directory %s already exists' % output_dir
                    sys.exit(1)
                
            intermediate_filename_root = '_trj' # A placeholder name
            jobs = []
            for i, clone_dir in enumerate(self.yield_xtc_directories(input_dir, input_style)):

                job = {'clone_dir' : clone_dir,
                       'output_dir': output_dir,
                       'pdb_file':   self.pdb_topology,
                       'trajectory_number': i,
                       'stride':     stride,
                       'max_rmsd':   max_rmsd,
                       'discard_first_N_snapshots': discard_first_N_snapshots,
                       'min_gens':   min_gens,
                       'center_conformations': center_conformations,
                       'memory_check': update
                       }
                jobs.append(job)  

            if num_proc > 1:
                # use multiprocessing
                pool = Pool(processes=num_proc)
                pool.map(self.write_trajectory_mapper, jobs)
            else:
                # use regular serial execution
                map(self.write_trajectory_mapper, jobs)

            # Rename trajectory files such that they have contiguous numbering
            print "\nFinished Generating Trajectories. Renaming them now in contiguous order"
            mapping = {} # document the directory changes, allowing us to update memory
            for i, filename in enumerate( os.listdir(output_dir) ):
                path = os.path.join(output_dir, filename)
                new_path = os.path.join(output_dir, "trj%d.lh5" % i)
                os.rename(path, new_path)
                mapping[path] = new_path

            # update the memory hash to accound for our renumbering
            for key in self.memory.keys():
                if key not in ['convert_parameters', 'SerializerFilename']:
                    print "%s --> %s" % ( self.memory[key][0], mapping[ self.memory[key][0] ] )
                    self.memory[key][0] = mapping[ self.memory[key][0] ]

            # save the parameters used for this run in the memory file, and write to disk
            print "\nGenerating Project File: %s" % self.projectinfo_file
            if update:
                try: os.remove( self.projectinfo_file ) # if we are updating, just start w fresh slate
                except: pass

            self.memory['convert_parameters'] = (input_dir, output_dir, stride, max_rmsd,
                                                 discard_first_N_snapshots, min_gens,
                                                 center_conformations, num_proc, self.projectinfo_file,
                                                 input_style )

            Project.CreateProjectFromDir( Filename         = self.projectinfo_file,
                                          TrajFilePath     = output_dir,
                                          TrajFileBaseName = 'trj',
                                          TrajFileType     = '.lh5',
                                          ConfFilename     = self.pdb_topology,
                                          initial_memory   = cPickle.dumps( self.memory ) )

            print "Data converted properly."

            return


        def update_trajectories(self):
            """ Using the memory state, updates a trajectory of LH5 Trajectory files by
            scanning a FAH project for new trajectories, and converting those """

            print self.memory['convert_parameters']
            (input_dir, output_dir, stride, max_rmsd, discard_first_N_snapshots, min_gens, \
            center_conformations, num_proc, self.projectinfo_file, input_style ) = self.memory['convert_parameters']
            
            self.write_all_trajectories(input_dir, output_dir, stride, max_rmsd,
                                        discard_first_N_snapshots, min_gens,
                                        center_conformations, num_proc, 
                                        input_style, update=True)
            
            return
        

        def write_trajectory_mapper(self, args):
            """
            Helper function:
            
            Because it is designed to be called via map(),
            which only supports supplying 1 iterable to the function, this function
            must formally only take 1 argument.
            """

            try:
                clone_dir  = args['clone_dir']
                output_dir = args['output_dir']
                trajectory_number = args['trajectory_number']
                stride = args['stride']
                max_rmsd = args['max_rmsd']
                discard_first_N_snapshots = args['discard_first_N_snapshots']
                min_gens = args['min_gens']
                center_conformations = args['center_conformations']
                memory_check = args['memory_check']
            except KeyError as e:
                print >> sys.stderr, """One or more required keys (%s) was not
                suplied in the input argument to create_trajectory().""" % e
                sys.exit(1)
                
            self.write_trajectory( clone_dir, output_dir, trajectory_number,
                                   stride, max_rmsd, discard_first_N_snapshots,
                                   min_gens, center_conformations, memory_check )
            
            return


        def write_trajectory( self, clone_dir, output_dir, trajectory_number,
                              stride, max_rmsd, discard_first_N_snapshots,
                              min_gens, center_conformations, memory_check ):
            """
            This function takes in a path to a CLONE and merges all the XTC files
            it finds into a LH5 trajectory:

            clone_dir: (string) the directory in which the xtc files are found. All
                       of the xtc files in this directory are joined together to make
                       a single trajectory (.lh5) output file
            output_dir: (string) directory where the outputted files will be placed
            trajectory_number: A unique number for this trajectory. This number
                               is used in constructing the filename to write the outputted .lh5
                               trajectory to, and thus must be unique
            stride: (integer) Subsample by only considering every Nth snapshop.
            max_rmsd: (integer/None) if this value is not None, calculate the RMSD to
                      the pdb_file from each snapshot and reject trajectories which have snapshots 
                      with RMSD greated than max_rmsd. If None, no check is performed
            discard_first_N_snapshots: (integer) ignore the first N snapshots of each trajectory
            min_gens: (integer) Discard the trajectories that contain fewer than N XTC files.
            center_conformations: (boolean) center conformations before saving.
            memory_check: (boolean) if yes, uses the memory dictionary to do an update rather
                          than a complete re-convert
            """
                
            xtc_files = self.list_xtcs_in_dir(clone_dir)

            # Ensure that we're only joining contiguously numbered xtc files -- starting at 0 --
            # into a trajectory. If there are gaps in the xtc files in the directory, we only
            # want to use the the ones such that they are contiguously numbered
            i = 0
            for i, filename in enumerate(xtc_files):
                if self.integer_component(filename) != i:
                    print "WARNING: Found discontinuity in xtc numbering - check data in %s" % clone_dir
                    xtc_files = xtc_files[0:i]
                    break
                

            # check the memory object to see which xtc files have already been converted, and
            # exclude those from this conversion
            if memory_check:
                if clone_dir in self.memory.keys():
                    previous_convert_exists = True
                    num_xtcs_converted = self.memory[clone_dir][1]
                    if len(xtc_files) == num_xtcs_converted: # if we have converted everything,
                        print "Already converted all files in %s, skipping..." % clone_dir
                        return                               # just bail out
                    else:
                        xtc_files = xtc_files[num_xtcs_converted:]
                else:
                    previous_convert_exists = False
            else:
                previous_convert_exists = False
    
            xtc_file_paths = [os.path.join(clone_dir, f) for f in xtc_files]
    
            print "Processing %d xtc files in clone_dir = %s" % (len(xtc_files), clone_dir)

            if len(xtc_files) <= min_gens:
                print "Skipping trajectory in clone_dir = %s" % clone_dir
                print "Too few xtc files (generations)."
                return

            try:
                trajectory = Trajectory.LoadFromXTC(xtc_file_paths, PDBFilename=self.pdb_topology)
            except IOError as e:
                print >> sys.stderr, "IOError (%s) when processing trajectory in clone_dir = %s" % (e, clone_dir)
                print >> sys.stderr, "Attempting rescue by disregarding final frame, which is often"
                print >> sys.stderr, "the first/only frame to be corrupted"

                if len(xtc_file_paths) == 1:
                    print "Didn't find any other frames in %s, continuing..." % clone_dir
                    return
                
                try:
                    trajectory = Trajectory.LoadFromXTC(xtc_file_paths[0:-1], PDBFilename=self.pdb_topology)
                except IOError:
                    print >> sys.stderr, "Unfortunately, the error remained even after ignoring the final frame."
                    print >> sys.stderr, "Skipping the trajectory in clone_dir = %s" % clone_dir
                    return
                else:
                    print >> sys.stderr, "Sucessfully recovered from IOError by disregarding final frame."

            if max_rmsd is not None:
                atomindices = [ int(i)-1 for i in trajectory['AtomID'] ]
                rmsdmetric = RMSD(atomindices)
                ppdb = rmsdmetric.prepare_trajectory(Trajectory.LoadTrajectoryFile(self.pdb_topology))
                ptraj = rmsdmetric.prepare_trajectory(trajectory)
                rmsds = rmsdmetric.one_to_all(ppdb, ptraj, 0)
            if max(rmsds) > max_rmsd:
                print >> sys.stderr, "Snapshot %d RMSD %f > the %f cutoff" % (argmax(rmsds), max(rmsds), max_rmsd)
                print >> sys.stderr, "Dropping trajectory"
                return

            if center_conformations:
                RMSD.TheoData.centerConformations(trajectory["XYZList"])

            # if we are adding to a previous trajectory, we have to load that traj up and extend it
            if previous_convert_exists:
                output_filename = self.memory[clone_dir][0]
                output_file_path = output_filename
                print "Extending: %s" % output_filename
                assert os.path.exists( output_filename )

                # load the traj and extend it
                Trajectory.AppendFramesToFile( output_filename, 
                                               trajectory['XYZList'][discard_first_N_snapshots::stride] )
                #old_trajectory = Trajectory.LoadTrajectoryFile( output_filename )
                #new_trajectory = old_trajectory.copy()
                #new_trajectory = Trajectory( new_trajectory )
                #new_trajectory['XYZList'] = concatenate( (old_trajectory['XYZList'],
                #    trajectory['XYZList'][discard_first_N_snapshots::stride]), axis=0 )

                ## remove the old trajectory from disk and write this one
                #os.remove( output_filename )
                #new_trajectory.Save(output_filename)

                num_xtcs_processed = len(xtc_file_paths) + self.memory[clone_dir][1]
                
            # if we are not adding to a traj, then we create a new one
            else:
                output_filename = 'trj%s.lh5' % trajectory_number
                output_file_path = os.path.join(output_dir, output_filename)

                if os.path.exists(output_file_path):
                    print >> sys.stderr, "The file name %s already exists. Skipping it." % output_file_path
                    return
            
                # stide and discard by snapshot
                trajectory['XYZList'] = trajectory['XYZList'][discard_first_N_snapshots::stride]
                trajectory.Save(output_file_path)

                num_xtcs_processed = len(xtc_file_paths)

            # log what we did into the memory object
            self.memory[clone_dir] = [ output_file_path, num_xtcs_processed ]

            return

        def yield_xtc_directories(self, input_dir, input_style):
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
                            #if filename.endswith('.xtc'):
                            #    return True
                            #else: return False
            
                            if contains_xtcs(filenames):
                                yield dirpath

                            else:
                                raise Exception(("""This function was invoked incorrectly.
                                                  The only acceptable choices for 'input_style' are
                                                  'FILE and 'FAH'. You supplied %s""" % input_style))


        def list_xtcs_in_dir(self, dir):
            pattern = re.compile('\D+(\d+)[.]xtc')
            xtc_files = [e for e in os.listdir(dir) if pattern.search(e)]
            xtc_files.sort(key=self.integer_component)
            print xtc_files
            return xtc_files


        def integer_component(self, filename):
            '''extract a the numeric part of the filename for sorting'''
            pattern = re.compile('\D+(\d+)[.]xtc')
            try:
                substr = pattern.match(filename).group(1)
                return int(substr)
            except:
                print >> sys.stderr,  "A filename (%s) may not have been sorted correctly" % filename
                return 0


    class _inject():
        """ Contains all of the methods for directing a FAH project. These
        functions will change what is on the workserver, modifying files
        and creating new files to manage sampling """


        # un-nest the local namespace from the parent class
        def __init__(self, outerclass):
            self.__dict__.update(outerclass.__dict__)


        def new_run(self):
            """ Creates a new run in the project directory, and adds that run
            to the project.xml file. Does not directly reboot the server """

            # create the new run directory
            raise NotImplementedError()


            # add the run to the project.xml

            return


        def stop_run(self, run, clone):
            """ Stops all CLONES in a RUN """
            raise NotImplementedError()


            return


        def stop_clone(self, run, clone):
            """ Stops the specified RUN/CLONE by changing the name of
            the WU's trr, adding .STOP to the end. """
            raise NotImplementedError()


            return

        
