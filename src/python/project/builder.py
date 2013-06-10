import os
import re
import logging
from glob import glob
from msmbuilder.utils import keynat
from msmbuilder import Trajectory
from msmbuilder.utils import keynat
import IPython
import numpy as np

from project import Project
from validators import ValidationError
logger = logging.getLogger(__name__)

class ProjectBuilder(object):
    def __init__(self, input_traj_dir, input_traj_ext, conf_filename, **kwargs):
        """
        Build an MSMBuilder project from a set of trajectories

        Parameters
        ----------
        input_traj_dir : str
            Root directory of the trajectory hierarchy. The trajectories should
            actually be in input_traj_dir/<something>/{files}.
        input_traj_ext : {'.xtc', '.dcd'}
            Trajectory file format
        conf_filename : str 
            path to a pdb
            this will be used to load xtc's and other trajectories

        Additional Parameters
        ---------------------
        stride : int
            Stride the input data at this frequency
        validators : [msmbuilder.project.Validator]
        project : msmbuilder.Project
            project instance that we are updating

        Attributes
        ----------
        project : msmbuilder.Project
            the built project

        Examples
        --------
        >>> pb = ProjectBuilder('XTC', '.xtc', 'native.pdb')
        >>> pb.project.save('ProjectInfo.yaml')
        """
        self.input_traj_dir = input_traj_dir
        self.input_traj_ext = input_traj_ext
        self.conf_filename = conf_filename

        self.output_traj_ext = '.lh5'
        self.project = kwargs.pop('project', None)

        if self.project is None:
            self.output_traj_basename = kwargs.pop('output_traj_basename', 'trj')
            self.output_traj_dir = kwargs.pop('output_traj_dir', 'Trajectories')
        else:
            self.output_traj_dir = os.path.relpath(os.path.dirname(self.project.traj_filename(0)))

            trj_name = os.path.basename(self.project._traj_paths[0])
            match_obj = re.search('(\w+)\d+\.lh5$', trj_name)
            if match_obj:
                self.output_traj_basename = match_obj.group(1)
            else:
                logger.warning("could not parse %s, defaulting to traj basename 'trj'", trj_name)
                self.output_traj_basename = 'trj'

        self.stride = kwargs.pop('stride', 1)
        self.project_updated = False
        # Keep track of when we've updated everything

        self._validators = []
        for e in kwargs.pop('validators', []):
            self.add_validator(e)

        if len(kwargs) > 0:
            raise ValueError('Unsupported arguments %s' % kwargs.keys())
        if input_traj_ext not in ['.xtc', '.dcd']:
            raise ValueError("Unsupported format")

        self._check_out_dir()

    def _validate_traj(self, traj):
        """
        Run the registered validators on  the trajectory

        Parameters
        ----------
        traj : msmbuilder.Trajectory

        Raises
        ------
        validation_error
            On failure
        """

        for validator in self._validators:
            validator(traj)

    def add_validator(self, validator):
        """
        Add a validator to the project builder
        
        Parameters
        ----------
        validator : callable
        
        Notes
        -----
        As the trajectories are being converted from their native format to
        MSMBuilder's format (lh5), all of the registeted validtors will be run
        against each trajectory. Validators should be callables like
        functions or classes with a __call__ method that check a
        trajectory. They are free to modify a trajectory as well, since it is
        passed by reference to the validator.
        
        If a validator detects a problem with a trajectory, it should raise
        a ValidationError -- that is, an error which subclasses
        msmbuilder.project.validators.ValidationError. When the ProjectBuilder
        detects a ValidationError, the error will be recorded in the project
        file, but the execution will procdede as normal and the trajectory will
        still be saved to disk. It will just be marked specially as "in error".
        
        In the current Project implementation, trajectories that are "in error"
        will be ignored -- when using project.load_traj(), only the "valid"
        trajectories will be returned, and project.n_trajs will only count
        the valid trajectories.
        """
        if not hasattr(validator, '__call__'):
            raise TypeError('Validator must be callable: %s' % validator)
        self._validators.append(validator)

    def get_project(self):
        """
        Retreive the project that was built.

        This can also be done with self.project

        Examples
        --------
        >>> pb = ProjectBuilder('XTC', '.xtc', 'native.pdb')
        >>> pb.convert()
        >>> pb.project == pb.get_project()
        True
        """
        if not self.project_updated:
            self.convert()
        return self.project

    def _check_out_dir(self):
        "Create self.output_traj_dir, or throw an error if it already exists"
        if not os.path.exists(self.output_traj_dir):
            os.makedirs(self.output_traj_dir)
        elif self.project is None:
            raise IOError('%s already exists' % self.output_traj_dir)
        else:  # project exists so we are supposed to be updating the 
               # trajectories
            pass

    def convert(self):
        """
        Main method for this class. Convert all of the trajectories into
        lh5 format and save them to self.output_traj_dir.

        Returns
        -------
        project : msmbuilder.projec
            The project object, summarizing the conversion
        """

        traj_lengths = []
        traj_paths = []
        traj_errors = []
        traj_converted_from = []

        if not self.project is None:
            old_traj_locs = ['/'.join([d for d in os.path.relpath(file_list[0]).split('/') if not d in ['.','..']][:-1]) for file_list in self.project._traj_converted_from]
            # ^^^ haha, @rmcgibbo, is this acceptable?
        else:
            old_traj_locs = []
        old_traj_locs = np.array(old_traj_locs)

        num_trajs_added = 0
        if self.project is None:
            num_orig_trajs = 0
        else:
            num_orig_trajs = len(self.project._traj_paths)
    
        for file_list in self._input_trajs():

            i = num_orig_trajs + num_trajs_added 
            # index for a new trajectory file is total we currently have (since it's zero-indexed)

            error = None

            traj_loc = '/'.join([d for d in os.path.relpath(file_list[0]).split('/') if not d in ['.', '..']][:-1])
            num_files = len(file_list)
            if not traj_loc in old_traj_locs:
                try:
                    traj = self._load_traj(file_list)
                    traj["XYZList"] = traj["XYZList"][::self.stride]
                except TypeError as e:
                    #traj_errors.append(e)
                    logger.warning('Could not convert %d files from %s (%s)', num_files, traj_loc, e)
                else:
                    num_trajs_added += 1
                    lh5_fn = os.path.join(self.output_traj_dir, 
                                      (self.output_traj_basename + str(i) + self.output_traj_ext))
                    traj.save(lh5_fn)
                    traj_lengths.append(len(traj["XYZList"]))
                    traj_paths.append(lh5_fn)
                    traj_converted_from.append(file_list)
            
                    try:
                        self._validate_traj(traj)
                        logger.info("%s (%d files), length %d, converted to %s", 
                                traj_loc, num_files, traj_lengths[-1], lh5_fn)
                    except ValidationError as e:
                        error = e
                        logger.error("%s (%d files), length %d, converted to %s with error '%s'", 
                                     traj_loc, num_files, traj_lengths[-1], lh5_fn, e)

                    traj_errors.append(error)

            else:  # Then this trajectory has been seen before, and so we need to 
                   # either extend it or skip it 
                   # This procedure would ideally be done using EArrays and we 
                   # use pyTables to actually extend the trajectory. Currently,
                   # the XYZList is a CArray, and so at some point we have to load
                   # the old array into memory, which is not the most efficient
                   # way to do this...
                old_ind = np.where(old_traj_locs == traj_loc)[0][0]
                old_locs = self.project._traj_converted_from[old_ind]
                old_num_files = len(old_locs)

                if old_num_files == len(file_list):
                    # Just assume if it is the same number of files then they are the
                    # same. We should change this eventually
                    logger.info("%s no change, did nothing for %s", traj_loc, self.project._traj_paths[old_ind])
                    continue
                elif old_num_files < len(file_list):
                    # Need to update the trajectory
                    try:
                        extended_traj = self._load_traj(file_list[old_num_files:])
                    except TypeError as e:
                        logger.warning('Could not convert: %s (%s)', file_list[old_num_files:], e)
                    else:
                        # assume that the first <old_num_files> are the same
                        # ^^^ This should be modified, but I want to get it working first
                        traj = self.project.load_traj(old_ind)
                        traj['XYZList'] = np.concatenate((traj['XYZList'], extended_traj['XYZList'][1:]))
                        # need to skip the first frame because this is what the xtc reader would do
                        traj.save(self.project._traj_paths[old_ind])
                        # This does what we want it to do, because msmbuilder.io.saveh
                        # deletes keys that exist already. However, this could be made more
                        # efficient by only updating the XYZList node since it's the
                        # only thing that changes

                        new_traj_locs = np.concatenate((old_locs, file_list[old_num_files:]))
                        self.project._traj_converted_from[old_ind] = list(new_traj_locs)
                        self.project._traj_lengths[old_ind] = len(traj['XYZList'])
                        # _errors updated later, saved to the same place, so traj_filename is the same
                        
                        try:
                            self._validate_traj(traj)
                            logger.info("%s (%d files), length %d, UPDATED %s",
                                        traj_loc, num_files, self.project._traj_lengths[old_ind], 
                                        self.project._traj_paths[old_ind])
                        except ValidationError as e:
                            error = e
                            logger.info("%s (%d files), length %d, UPDATED %s",
                                        traj_loc, num_files, self.project._traj_lengths[old_ind], 
                                        self.project._traj_paths[old_ind])

                            if self.project._traj_errors[old_ind] is None:
                                self.project._traj_errors[old_ind] = error
                            elif isinstance(self.project._traj_errors[old_ind], list):
                                self.project._traj_errors[old_ind].append(error)
                            else:
                                self.project._traj_errors[old_ind] = [self.project._traj_errors[old_ind], error]

                else:
                    # Somehow lost some frames...
                    logger.warn('Fewer frames found than currently have. Skipping. (%s)' % traj_loc)
                    continue

        if len(traj_paths) == 0 and self.project is None:
            os.rmdir(self.output_traj_dir)
            raise RuntimeError('No conversion jobs found!')

        if not self.project is None:
            traj_lengths = list(self.project._traj_lengths) + traj_lengths
            traj_paths = list(self.project._traj_paths) + traj_paths
            traj_errors = list(self.project._traj_errors) + traj_errors
            traj_converted_from = list(self.project._traj_converted_from) + traj_converted_from
            traj_converted_from = [[str(i) for i in l] for l in traj_converted_from]

        self.project = Project({'conf_filename': self.conf_filename,
                                'traj_lengths': traj_lengths,
                                'traj_paths': traj_paths,
                                'traj_errors': traj_errors,
                                'traj_converted_from': traj_converted_from})

        self.updated_project = True

        #IPython.embed()


    def _input_trajs(self):
        logger.warning("WARNING: Sorting trajectory files by numerical values in their names.")
        logger.warning("Ensure that numbering is as intended.")

        traj_dirs = glob(os.path.join(self.input_traj_dir, "*"))
        traj_dirs.sort(key=keynat)
        logger.info("Found %s traj dirs", len(traj_dirs))
        for traj_dir in traj_dirs:
            to_add = glob(traj_dir + '/*'+ self.input_traj_ext)
            to_add.sort(key=keynat)
            if to_add:
                yield to_add

    def _load_traj(self, file_list):
        """
        Load a set of xtc or dcd files as a single trajectory

        Note that the ordering of `file_list` is relevant, as the trajectories
        are catted together.

        Returns
        -------
        traj : msmbuilder.Trajectory
        """

        if self.input_traj_ext == '.xtc':
            traj = Trajectory.load_from_xtc(file_list, PDBFilename=self.conf_filename,
                        discard_overlapping_frames=True)
        elif self.input_traj_ext == '.dcd':
            traj = Trajectory.load_from_dcd(file_list, PDBFilename=self.conf_filename)
        else:
            raise ValueError()
        return traj


class FahProjectBuilder(ProjectBuilder):
    """
    Build a project using the classic FAH-style directory setup. E.g. looks
    for data of the form
    -- RUNs
    ---- CLONEs
    ------ frame0.xtc, frame1.xtc ....
    
    Contains a little more forgiving code than the standard ProjectBuilder,
    specifically if a certain CLONE will not load, then we try to load again
    excluding the last frame (which is often a FAH crash). This helps out
    quite a bit.

    Parameters
    ----------
    input_traj_dir : str
        Root directory of the trajectory hierarchy. The trajectories should
        actually be in input_traj_dir/<something>/{files}.
    input_traj_ext : {'.xtc', '.dcd'}
        Trajectory file format
    conf_filename : str
        Path to a pdb

    Additional Parameters
    ---------------------
    stride : int
        Stride the input data at this frequency
    validators : [msmbuilder.project.Validator]

    Attributes
    ----------
    project : msmbuilder.Project
        the built project

    Examples
    --------
    >>> pb = ProjectBuilder('XTC', '.xtc', 'native.pdb')
    >>> pb.project.save('ProjectInfo.yaml')
    
    """
    

    def _input_trajs(self):
        
        run_dirs = glob(os.path.join(self.input_traj_dir, "RUN*"))
        run_dirs.sort(key=keynat)
        logger.info("Found %d RUN dirs", len(run_dirs))
        
        for run_dir in run_dirs:
            clone_dirs = glob(os.path.join(run_dir, "CLONE*"))
            logger.info("%s: Found %d CLONE dirs", run_dir, len(clone_dirs))

            for clone_dir in clone_dirs:
                to_add = glob(clone_dir + '/*'+ self.input_traj_ext)
                to_add.sort(key=keynat)
                if to_add:
                    yield to_add
    
    
    def _load_traj(self, file_list):
        traj = None
        
        try:
            traj = super(FahProjectBuilder, self)._load_traj(file_list)
        except IOError as e:
            if e.errno == 2: # Then the pdb filename doesn't exist
                raise e
            corrupted_files = True
            n_corrupted = 1
            logger.error("Some files appear to be corrupted")
            while corrupted_files:
                logger.error("Trying to recover by discarding the %d-th-to-last file", n_corrupted)
                try:
                    traj = super(FahProjectBuilder, self)._load_traj(file_list[:-n_corrupted])
                except IOError:
                    n_corrupted += 1
                else:
                    logger.error("That seemed to work")
                    corrupted_files = False
        
        if traj is None:
            raise RuntimeError("Corrupted frames in %s, recovery impossible" % file_list)
            
        return traj
    

