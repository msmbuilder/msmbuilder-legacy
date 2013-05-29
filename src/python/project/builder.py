import os
import logging
from glob import glob
from msmbuilder.utils import keynat
from msmbuilder import Trajectory
from msmbuilder.utils import keynat

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
        self.input_traj_dir = input_traj_dir
        self.input_traj_ext = input_traj_ext
        self.conf_filename = conf_filename
        self.project = None

        self.output_traj_ext = '.lh5'
        self.output_traj_basename = kwargs.pop('output_traj_basename', 'trj')
        self.output_traj_dir = kwargs.pop('output_traj_dir', 'Trajectories')
        self.stride = kwargs.pop('stride', 1)

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

        for valilator in self._validators:
            valilator(traj)

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
        if self.project is None:
            self.convert()
        return self.project

    def _check_out_dir(self):
        "Create self.output_traj_dir, or throw an error if it already exists"
        if not os.path.exists(self.output_traj_dir):
            os.makedirs(self.output_traj_dir)
        else:
            raise IOError('%s already exists' % self.output_traj_dir)

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

        for i, file_list in enumerate(self._input_trajs()):
            
            error = None
            
            try:
                traj = self._load_traj(file_list)
                traj["XYZList"] = traj["XYZList"][::self.stride]
            except TypeError as e:
                traj_errors.append(e)
                logger.warning('Could not convert: %s (%s)', file_list, e)
            else:
                lh5_fn = os.path.join(self.output_traj_dir, 
                                      (self.output_traj_basename + str(i) + self.output_traj_ext))
                traj.save(lh5_fn)
                traj_lengths.append(len(traj["XYZList"]))
                traj_paths.append(lh5_fn)
                traj_converted_from.append(file_list)
            
                try:
                    self._validate_traj(traj)
                    logger.info("%s, length %d, converted to %s", 
                                file_list, traj_lengths[-1], lh5_fn)
                except ValidationError as e:
                    error = e
                    logger.error("%s, length %d, converted to %s with error '%s'", 
                                 file_list, traj_lengths[-1], lh5_fn, e)

            traj_errors.append(error)

        if len(traj_paths) == 0:
            os.rmdir(self.output_traj_dir)
            raise RuntimeError('No conversion jobs found!')


        self.project = Project({'conf_filename': self.conf_filename,
                                'traj_lengths': traj_lengths,
                                'traj_paths': traj_paths,
                                'traj_errors': traj_errors,
                                'traj_converted_from': traj_converted_from})

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
    

