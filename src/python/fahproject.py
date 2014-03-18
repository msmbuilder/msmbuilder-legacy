"""
This is a class that contains all of the necessary tools to interact
with folding@home projects.

Written by: TJ Lane <tjlane@stanford.edu>
Contributions from Robert McGibbon
"""
from __future__ import print_function, division, absolute_import
# GLOBAL IMPORTS
import os
import re
import sys
import cPickle
import time
from glob import glob

import smtplib
from email.mime.text import MIMEText

from numpy import argmax

import subprocess
from subprocess import PIPE

from multiprocessing import Pool
try:
    from deap import dtm
except:
    pass

from msmbuilder import Trajectory
from msmbuilder.metrics import RMSD
from msmbuilder.project import Project
from msmbuilder.utils import make_methods_pickable, keynat
make_methods_pickable()
import logging
logger = logging.getLogger(__name__)

class FahProject(object):
    """
    A generic class for interacting with Folding@home projects

    Parameters
    ----------
    pdb : str
        The pdb file on disk associated with the project.

    project_number: int
        The project number assocaited with the project.

    projectinfo_file : str
        Name of the project info file.

    work_server : str
        Hostname of the work server to interact with.

    email : str
        email to forward alerts to
    """

    def __init__(self, pdb, project_number=1, projectinfo_file="ProjectInfo.h5", 
                 work_server=None, email=None):

        # metadata associated with a FAH project
        self.project_number   = project_number
        self.work_server      = work_server
        self.pdb_topology     = pdb
        self.manager_email    = email
        self.projectinfo_file = projectinfo_file

		# check that the PDB exists
        if not os.path.exists(self.pdb):
            logger.error("Cannot find %s", self.pdb)

        # load in the memory state
        if os.path.exists( projectinfo_file ):
            self.load_memory_state( projectinfo_file )
        else:
            self.memory = {}
            logger.info("No file: %s found. Generating new memory state.", projectinfo_file)
            logger.info("Processing all trajectories, will save progress to: %s", projectinfo_file)

        # set the nested classes defined below to be separate namespaces
        # this should separate concerns and prevent (dangerous) user mistakes
        self.retrieve = _retrieve(self)
        self.inject   = _inject(self)


    def restart_server(self):
        """
        Restarts the workserver, should be called when injecting runs.

        Checks that the server comes back up without throwing an error -
        if it doesn't come up OK, sends mail to the project manager.
        """

        raise NotImplementedError()

        # restart the server, wait 60s to let it come back up
        logger.warning("Restarting server: %s", self.work_server)
        stop_cmd  = "/etc/init.d/FAHWorkServer-%s stop" % self.work_server
        start_cmd = "/etc/init.d/FAHWorkServer-%s start" % self.work_server
        r = subprocess.call(stop_cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        time.sleep(60)
        r = subprocess.call(stop_cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)

        # check that we came back up OK, if not freak out
        processname = "FAHWorkServer-%s" % self.work_server
        is_alive = False # guilty until proven innocent

        for line in os.popen("ps -a"):
            if line.find(processname) > 0:
                is_alive = True

        if not is_alive:
            error_msg = """
                    FATAL ERROR: msmbuilder.FahProject is reporting a critical issue:

                             Workserver %s did not come back up after restart!

                    Recommend you attend to this immediately.""" % self.workserver

            if email: send_error_email(self, error_msg)
            raise Exception(error_msg)

        return



    def send_error_email(self, error_msg):
        """
        Sends an error message to the registered email.

        Parameters
        ----------
        error_msg : str
            The string to include in the email.
        """

        raise NotImplementedError()

        if email == None:
            logger.error("Cannot send error email - no email provided")
            return

        msg = MIMEText(error_msg)

        msg['Subject'] = '[msmbuilder.FahProject] FATAL ERROR IN FAHPROJECT'
        msg['From'] = 'msmbuilder@gmail.com'
        msg['To'] = self.email

        # Send the message via our own SMTP server, but don't include the envelope header.
        logger.error("Sending error email to: %s", self.email)
        s = smtplib.SMTP('smtp.gmail.com')
        s.sendmail(me, [you], msg.as_string())
        s.quit()

        return


    def save_memory_state(self):
        """
        Saves the 'memory state' to disk in a serialized format.

        Notes
        -----
        When saving, we encode the keys into base16 with a leading 'a',
        because the HDF5 Serializer doesnt' like '/' characters and is
        super picky in general

        See Also
        --------
        load_memory_state
        """

        project_info = Project.load_from_hdf( projectinfo_file )
        project_info["Memory"] = cPickle.dumps( self.memory )
        project_info.save_to_hdf( projectinfo_file, do_file_check=False )

        return


    def load_memory_state(self, projectinfo_file):
        """
        Loads the 'memory state' from a serialized file on disk

        Parameters
        ----------
        projectinfo_file : str
            The file on disk from which to read.

        Notes
        -----
        When reading the memory state, we have to decode base16, and
        also remove leading 'a' characters

        See Also
        --------
        save_memory_state
        """

        logger.info("Loading memory state from: %s", projectinfo_file)

        project_info = Project.load_from_hdf( projectinfo_file )
        self.memory = cPickle.loads( project_info["Memory"] )

        return


class _inject(object):
    """
    Contains all of the methods for directing a FAH project. These
    functions will change what is on the workserver, modifying files
    and creating new files to manage sampling.
    """

    #################################################################
    #
    # The below code is still under development. We are waiting on a
    # few more things to progress in the FAH WS code API, and also
    # advances in MSMAccelerator, which will make these methods useful.
    # -- TJL, Aug. '12
    #

    # un-nest the local namespace from the parent class
    def __init__(self, outerclass):
        """
        Initialize the _inject subclass of FahProject. Should not
        be called by the user!
        """

        raise NotImplementedError('If youd like to use this functionality, email TJ'
    								  '<tjlane@stanford.edu> and send love and affection')

        self.__dict__.update(outerclass.__dict__)
        self.send_error_email = outerclass.send_error_email
        self.restart_server = outerclass.restart_server

        # if we are on a workserver,
        if self.work_server != None:
            self.set_project_basepath()


    def set_project_basepath(self):
        """
        Finds and internally stores a FAH Project's path.
        """
        search = glob("/home/*/server2/data/SVR*/PROJ%d" % self.project_number)
        if len(search) != 1:
            raise Exception("Could not find unique FAH project: %d on %s" % (self.project_number,
                                                                             self.work_server))
        else: self.project_basepath = search[0]


    def new_run(self):
        """
        Creates a new run in the project directory, and adds that run
        to the project.xml file. Does not directly reboot the server.
        """

        # create the new run directory
        raise NotImplementedError()
        # add the run to the project.xml


    def stop_run(self, run):
        """
        Stops all CLONES in a RUN.

        Parameters
        ----------
        run : int
            The run to stop.
        """

        logger.warning("Shutting down RUN%d", run)
        clone_dirs = glob(run_dir + "CLONE*")
        for clone_dir in clone_dirs:
            g = re.search('CLONE(\d+)', 'CLONE55')
            if g:
                clone = g.group(1)
                self.stop_clone(run, clone)

        return


    def stop_clone(self, run, clone):
        """
        Stops the specified RUN/CLONE by changing the name of
        the WU's trr, adding .STOP to the end.

        Parameters
        ----------
        run : int
            The run containing the clone to stop.

        clone : int
            The clone to stop.
        """

        clone_dir = os.path.join(self.project_basepath, 'RUN%d/' % run, 'CLONE%d/' % clone)

        # add .STOP to all dem TRR files
        trrs = glob( clone_dir + '*.trr' )
        if len(trrs) == 0:
            logger.error("Could not find any TRRs to stop in %s. Proceeding.", clone_dir)
        else:
            for trr in trrs:
                os.rename(trr, trr+'.STOP')
                loggger.info("Stopped: %s", trr)

        return




class _retrieve(object):
    """
    Contains all of the functions necessary for getting data from
    the workserver. Functions are able to navagate the WS directory
    structure and get trajectories and other information.
    """

    # un-nest the local namespace from the parent class
    def __init__(self, outerclass):
        """
        Initialize the _retrieve subclass of FahProject. Should not
        be called by the user!
        """
        self.__dict__.update(outerclass.__dict__)
        self.save_memory_state = outerclass.save_memory_state


    def write_all_trajectories(self, input_dir, output_dir, stride, max_rmsd,
                               min_gens, center_conformations, num_proc,
                               input_style, update=False):
        """
        Convert all of the trajectories in the FAH project in input_dir to
        h5 trajectory files which will be placed in output dir.

        If the 'update' flag is set, then will use the memory object to check for
        previously converted data, and add to it (rather than reconverting everything).
        This functionality can be more cleanly called through the update_trajectories()
        method.

        Parameters
        ----------
        input_dir : str
            The directory to look for XTC/DCD files in.

        output_dir : str
            The place to write the converted h5s

        stride : int
            The size of the stride to employ. E.g., if stride = 3, the script
            keeps every 3rd MD snapshot from the original data. Useful to throw
            away highly correlated data if snapshots were saved frequently.

        max_rmsd : float
            Throw away any data that is further than `max_rmsd` (in nm) from the
            pdb file associated with the project. This is used as a sanity check
            to prevent including, e.g. data from a simulation that is blowing up.

        min_gens : int
            Discard trajectories with fewer than `min_gens` generations.

        center_conformations : bool
            Whether to center the converted (h5) conformations.

        num_proc : int
            Number of processors to employ. Note that this function is typically
            I/O limited, so paralellism is unlikely to yield much gain.

        input_style : {'FAH', 'FILE'}
            If you use input_style = 'FAH', this code uses knowledge of the
            RUN*/CLONE* directory structure to yield all the CLONE directories.
            If you use input_style = 'FILE', this code uses os.walk() which is
            A LOT slower because it has to stat every file, but is capable of
            recursively searching for xtc files to arbitrary depths.

        update : bool
            If `True`, then tries to figure out what data has already been converted
            by reading the "memory state" in the provided ProjectInfo file, and only
            converts new data. If `False`, does a fresh re-convert.


        Notes
        -----
        Since sometimes a conversion fails, we collect all trajectories at the
        end and renumber them such that they are contiguously numbered.
        """

        if update:
            assert os.path.exists(output_dir)
        else:
            try:
                os.mkdir(output_dir)
            except OSError:
                logger.error('Error: The directory %s already exists', output_dir)
                sys.exit(1)

        intermediate_filename_root = '_trj' # A placeholder name

        #dtm does not play nice with OpenMP
        use_parallel_rmsd = (num_proc != 'use_dtm_instead')

        jobs = []
        for i, clone_dir in enumerate(self.yield_xtc_directories(input_dir, input_style)):

            job = {'clone_dir' : clone_dir,
                   'output_dir': output_dir,
                   'pdb_file':   self.pdb_topology,
                   'trajectory_number': i,
                   'stride':     stride,
                   'max_rmsd':   max_rmsd,
                   'min_gens':   min_gens,
                   'center_conformations': center_conformations,
                   'memory_check': update,
                   'omp_parallel_rmsd': use_parallel_rmsd
                   }
            jobs.append(job)

        if len(jobs) == 0:
            raise RuntimeError('No conversion jobs found!')

        if num_proc == 'use_dtm_instead':
            # use DTM mpi parallel map
            dtm.map(self.write_trajectory_mapper, jobs)
        elif num_proc > 1:
            # use multiprocessing
            pool = Pool(processes=num_proc)
            pool.map(self.write_trajectory_mapper, jobs)
        else:
            # use regular serial execution
            for j in jobs:
                self.write_trajectory_mapper(j)

        # Rename trajectory files such that they have contiguous numbering
        logger.info("Finished Generating Trajectories. Renaming them now in contiguous order")
        mapping = {} # document the directory changes, allowing us to update memory
        for i, filename in enumerate( sorted( os.listdir(output_dir), key=keynat) ):
            path = os.path.join(output_dir, filename)
            new_path = os.path.join(output_dir, "trj%d.h5" % i)
            os.rename(path, new_path)
            mapping[path] = new_path

        # update the memory hash to accound for our renumbering
        for key in self.memory.keys():
            if key not in ['convert_parameters', 'SerializerFilename']:
                logger.info("%s --> %s", self.memory[key][0], mapping[ self.memory[key][0] ])
                self.memory[key][0] = mapping[ self.memory[key][0] ]

        # save the parameters used for this run in the memory file, and write to disk
        logger.info("Generating Project File: %s", self.projectinfo_file)
        if update:
            try: os.remove( self.projectinfo_file ) # if we are updating, just start w fresh slate
            except: pass

        self.memory['convert_parameters'] = (input_dir, output_dir, stride, max_rmsd, min_gens,
                                             center_conformations, num_proc, self.projectinfo_file,
                                             input_style )

        Project.CreateProjectFromDir( Filename         = self.projectinfo_file,
                                      TrajFilePath     = output_dir,
                                      TrajFileBaseName = 'trj',
                                      TrajFileType     = '.h5',
                                      ConfFilename     = self.pdb_topology,
                                      initial_memory   = cPickle.dumps( self.memory ) )

        logger.info("Data converted properly.")

        return


    def update_trajectories(self):
        """
		Using the memory state, updates a trajectory of H5 Trajectory files by
        scanning a FAH project for new trajectories, and converting those.
		"""

        logger.info(self.memory['convert_parameters'])
        (input_dir, output_dir, stride, max_rmsd, min_gens, \
        center_conformations, num_proc, self.projectinfo_file, input_style ) = self.memory['convert_parameters']

        self.write_all_trajectories(input_dir, output_dir, stride, max_rmsd, min_gens,
                                    center_conformations, num_proc, input_style, update=True)

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
            min_gens = args['min_gens']
            center_conformations = args['center_conformations']
            memory_check = args['memory_check']
            omp_parallel_rmsd = args['omp_parallel_rmsd']
        except KeyError as e:
            logger.critical("""One or more required keys (%s) was not
            suplied in the input argument to create_trajectory().""", e)
            sys.exit(1)

        self.write_trajectory(clone_dir, output_dir, trajectory_number, stride, max_rmsd,
                              min_gens, center_conformations, memory_check, omp_parallel_rmsd)

        return


    def write_trajectory(self, clone_dir, output_dir, trajectory_number, stride,
						 max_rmsd, min_gens, center_conformations, memory_check,
						 omp_parallel_rmsd=True):
        """
        This function takes in a path to a CLONE and merges all the XTC files
        it finds into a H5 trajectory:

        Parameters
        ----------
        clone_dir : str
            the directory in which the xtc files are found. All of the xtc files
            in this directory are joined together to make a single trajectory
            (.h5) output file

        output_dir : str
            directory where the outputted files will be placed

        trajectory_number : int
            A unique number for this trajectory. This number is used in
            constructing the filename to write the outputted .h5 trajectory to,
            and thus must be unique

        stride: int
            Subsample by only considering every Nth snapshop.
        max_rmsd: {int, None}
            if this value is not None, calculate the RMSD to the pdb_file from
            each snapshot and reject trajectories which have snapshots with RMSD
            greated than max_rmsd. If None, no check is performed

        min_gens : int
            Discard the trajectories that contain fewer than `min_gens` XTC files.

        center_conformations : bool
            center conformations before saving.

        memory_check : bool
            if yes, uses the memory dictionary to do an update rather than a
            complete re-convert.

        omp_parallel_rmsd : bool
            If true, use OpenMP accelerated RMSD calculation for max_rmsd check
        """

        xtc_files = self.list_xtcs_in_dir(clone_dir)

        # Ensure that we're only joining contiguously numbered xtc files -- starting at 0 --
        # into a trajectory. If there are gaps in the xtc files in the directory, we only
        # want to use the the ones such that they are contiguously numbered
        i = 0
        for i, filename in enumerate(xtc_files):
            if self.integer_component(filename) != i:
                logger.error("Found discontinuity in xtc numbering - check data in %s", clone_dir)
                xtc_files = xtc_files[0:i]
                break


        # check the memory object to see which xtc files have already been converted, and
        # exclude those from this conversion
        if memory_check:
            if clone_dir in self.memory.keys():
                previous_convert_exists = True
                num_xtcs_converted = self.memory[clone_dir][1]
                if len(xtc_files) == num_xtcs_converted: # if we have converted everything,
                    logger.info("Already converted all files in %s, skipping...", clone_dir)
                    return                               # just bail out
                else:
                    xtc_files = xtc_files[num_xtcs_converted:]
            else:
                previous_convert_exists = False
        else:
            previous_convert_exists = False

        xtc_file_paths = [os.path.join(clone_dir, f) for f in xtc_files]

        logger.info("Processing %d xtc files in clone_dir = %s", len(xtc_files), clone_dir)

        if len(xtc_files) <= min_gens:
            logger.info("Skipping trajectory in clone_dir = %s", clone_dir)
            logger.info("Too few xtc files (generations).")
            return

        try:
            # [this should check for and discard overlapping snapshots]
            trajectory = Trajectory.load_from_xtc(xtc_file_paths, PDBFilename=self.pdb_topology,
                                                discard_overlapping_frames=True)
        except IOError as e:
            logger.error("IOError (%s) when processing trajectory in clone_dir = %s", e, clone_dir)
            logger.error("Attempting rescue by disregarding final frame, which is often")
            logger.error("the first/only frame to be corrupted")

            if len(xtc_file_paths) == 1:
                logger.error("Didn't find any other frames in %s, continuing...", clone_dir)
                return

            try:
                trajectory = Trajectory.load_from_xtc(xtc_file_paths[0:-1], PDBFilename=self.pdb_topology)
            except IOError:
                logger.error("Unfortunately, the error remained even after ignoring the final frame.")
                logger.error("Skipping the trajectory in clone_dir = %s", clone_dir)
                return
            else:
                logger.error("Sucessfully recovered from IOError by disregarding final frame.")

        if max_rmsd is not None:
            atomindices = [ int(i)-1 for i in trajectory['AtomID'] ]
            rmsdmetric = RMSD(atomindices, omp_parallel=omp_parallel_rmsd)
            ppdb = rmsdmetric.prepare_trajectory(Trajectory.load_trajectory_file(self.pdb_topology))
            ptraj = rmsdmetric.prepare_trajectory(trajectory)
            rmsds = rmsdmetric.one_to_all(ppdb, ptraj, 0)

            if max(rmsds) > max_rmsd:
                logger.warning("Snapshot %d RMSD %f > the %f cutoff" , argmax(rmsds), max(rmsds), max_rmsd)
                logger.warning("Dropping trajectory")
                return

        if center_conformations:
            RMSD.TheoData.centerConformations(trajectory["XYZList"])

        # if we are adding to a previous trajectory, we have to load that traj up and extend it
        if previous_convert_exists:
            output_filename = self.memory[clone_dir][0]
            output_file_path = output_filename
            logger.info("Extending: %s", output_filename)
            assert os.path.exists( output_filename )

            # load the traj and extend it [this should check for and discard overlapping snapshots]
            Trajectory.append_frames_to_file( output_filename,
                                           trajectory['XYZList'][::stride],
                                           discard_overlapping_frames=True )

            num_xtcs_processed = len(xtc_file_paths) + self.memory[clone_dir][1]

        # if we are not adding to a traj, then we create a new one
        else:
            output_filename = 'trj%s.h5' % trajectory_number
            output_file_path = os.path.join(output_dir, output_filename)

            if os.path.exists(output_file_path):
                logger.info("The file name %s already exists. Skipping it.", output_file_path)
                return

            # stide and discard by snapshot
            trajectory['XYZList'] = trajectory['XYZList'][::stride]
            trajectory.save(output_file_path)

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

        Parameters
        ----------
        input_dir : str
            The directory to read xtcs from

        input_style : {'FILE', 'FAH'}
            Which search strategy to employ.
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
        """
        Find all the xtc files in `dir`.

        Parameters
        ----------
        dir : str
            Path of the directory to look in.

        Returns
        -------
        xtc_files : list
            List of the xtcs in `dir`.
        """
        pattern = re.compile('\D+(\d+)[.]xtc')
        xtc_files = [e for e in os.listdir(dir) if pattern.search(e)]
        xtc_files.sort(key=self.integer_component)
        logger.info(xtc_files)
        return xtc_files


    def integer_component(self, filename):
        """
        Extract a the numeric part of the filename for sorting

        Parameters
        ----------
        filename : str
           The file name to parse.

        Returns
        -------
        substr : str
            The numeric part of the filename for sorting.
        """
        pattern = re.compile('\D+(\d+)[.]xtc')
        try:
            substr = pattern.match(filename).group(1)
            return int(substr)
        except:
            logger.error("A filename (%s) may not have been sorted correctly", filename)
            return 0
