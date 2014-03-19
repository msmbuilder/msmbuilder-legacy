from __future__ import print_function, absolute_import, division

import os
import re
import logging
from glob import glob
from msmbuilder.utils import keynat
import numpy as np
import mdtraj as md

from .project import Project
from .validators import ValidationError
logger = logging.getLogger(__name__)


class ProjectBuilder(object):

    def __init__(self, input_traj_dir, input_traj_ext, conf_filename,
                 stride=1, project=None, validators=[], output_traj_dir='Trajectories',
                 output_traj_ext='.h5', output_traj_basename='trj', atom_indices=None):
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
        stride : int, optional
            Stride the input data at this frequency
        validators : list, optional
            list of validators
        project : msmbuilder.Project or None
            project instance that we are updating
        output_traj_dir : str, optional
            output directory to save trajectories
        output_traj_ext : str, optional
            output extension for trajectories
        output_traj_basename : str, optional
            output trajectory basename
        atom_indices : np.ndarray
            Select only these atom indices when loading trajectories and PDBs
            (Zero-based index).  If None, selected all atoms.  This also will
            cause a new PDB to be written with name conf_filename.subset.pdb

        Attributes
        ----------
        project : msmbuilder.Project
            the built project

        Examples
        --------
        >>> pb = ProjectBuilder('XTC', '.xtc', 'native.pdb')
        >>> pb.project.save('ProjectInfo.yaml')
        """
        self.atom_indices = atom_indices
        self.input_traj_dir = input_traj_dir.strip()
        self.input_traj_ext = input_traj_ext.strip()
        if self.input_traj_ext[0] != '.':
            self.input_traj_ext = '.%s' % self.input_traj_ext
        self.conf_filename = conf_filename.strip()

        # If not using atom_indices, we work exclusively with the original conf_filename
        if self.atom_indices is None:
            self.conf_filename_final = self.conf_filename
        else:
            # If we are selecting atom subsets, we need to generate
            # a new PDB file with the desired atoms, which will be called
            # conf_filename.subset.pdb
            filename_pieces = list(os.path.splitext(self.conf_filename))
            filename_pieces.insert(-1, ".subset")
            self.conf_filename_final = "".join(filename_pieces)
            subset_conf = md.load(self.conf_filename, atom_indices=atom_indices)
            subset_conf.save(self.conf_filename_final)

        self.conf = md.load(self.conf_filename)

        self.output_traj_ext = output_traj_ext.strip()
        if self.output_traj_ext[0] != '.':
            self.output_traj_ext = '.%s' % self.output_traj_ext
        self.project = project

        if self.project is None:
            self.output_traj_basename = output_traj_basename.strip()
            self.output_traj_dir = output_traj_dir.strip()
        else:
            logger.info("using project data to determine trajectory basename/location/extension")
            self.output_traj_dir = os.path.relpath(os.path.dirname(self.project.traj_filename(0)))

            trj_name = os.path.basename(self.project._traj_paths[0])
            match_obj = re.search('(\w+)\d+\.%s$' % self.output_traj_ext[1:], trj_name)
            if match_obj:
                self.output_traj_basename = match_obj.group(1)
            else:
                logger.warning("could not parse %s, defaulting to traj basename 'trj'", trj_name)
                self.output_traj_basename = 'trj'

        self.stride = int(stride)
        self.project_updated = False
        # Keep track of when we've updated everything

        # setup containers for all of the metadata
        self.traj_lengths = []
        self.traj_errors = []
        self.traj_converted_from = []
        self.traj_paths = []

        self._validators = []
        for e in validators:
            self.add_validator(e)

        if input_traj_ext not in ['.xtc', '.dcd']:
            raise ValueError("Unsupported format")

        self._check_out_dir()

    def _validate_traj(self, traj):
        """
        Run the registered validators on  the trajectory

        Parameters
        ----------
        traj : mdtraj.Trajectory

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
        file, but the execution will procede as normal and the trajectory will
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

    def _load_new_files(self, file_list, traj_loc):
        """
        load a new trajectory specified by a list of input files.
        """

        num_files = len(file_list)
        try:
            traj, n_loaded = self._load_traj(file_list)
            traj = traj[::self.stride]
        except RuntimeError as e:
            # traj_errors.append(e)
            logger.warning('Could not convert %d files from %s (%s)', num_files, traj_loc, e)
        else:
            if self.project is None:
                next_ind = len(self.traj_lengths)
            else:
                next_ind = len(self.traj_lengths) + len(self.project._traj_lengths)

            out_fn = os.path.join(self.output_traj_dir,
                                  (self.output_traj_basename + str(next_ind) + self.output_traj_ext))
            traj.save(out_fn)
            self.traj_lengths.append(traj.n_frames)
            self.traj_paths.append(out_fn)
            self.traj_converted_from.append(file_list[:n_loaded])

            error = None
            try:
                self._validate_traj(traj)
                logger.info("%s (%d files), length %d, converted to %s",
                            traj_loc, n_loaded, self.traj_lengths[-1], out_fn)
            except ValidationError as e:
                error = e
                logger.error("%s (%d files), length %d, converted to %s with error '%s'",
                             traj_loc, num_files, self.traj_lengths[-1], out_fn, e)

            self.traj_errors.append(error)

    def _update_traj(self, old_ind, file_list, traj_loc):
        """
        update a trajectory that we already had in the old project.
        """
        # This trajectory has been seen before, and so we need to
        # either extend it or skip it
        # This procedure would ideally be done using EArrays and we
        # use pyTables to actually extend the trajectory. Currently,
        # the XYZList is a CArray, and so at some point we have to load
        # the old array into memory, which is not the most efficient
        # way to do this...
        old_locs = self.project._traj_converted_from[old_ind]
        old_num_files = len(old_locs)
        num_files = len(file_list)

        if old_num_files == len(file_list):
            # Just assume if it is the same number of files then they are the
            # same. We should change this eventually
            logger.info("%s no change, did nothing for %s",
                        traj_loc, self.project._traj_paths[old_ind])

        elif old_num_files < len(file_list):
            # Need to update the trajectory
            try:
                extended_traj, n_loaded = self._load_traj(file_list[old_num_files:])
            except RuntimeError as e:
                logger.warning('Could not convert: %s (%s)', file_list[old_num_files:], e)
            else:
                # assume that the first <old_num_files> are the same
                # ^^^ This should be modified, but I want to get it working first
                traj = self.project.load_traj(old_ind)

                # remove the redundant first frame if it is actually redundant
                if np.abs(traj.xyz[-1] - extended_traj.xyz[0]).sum() < 1E-8:
                    extended_traj = extended_traj[1:]

                traj.xyz = np.concatenate((traj.xyz, extended_traj.xyz))

                traj.save(self.project._traj_paths[old_ind])
                # This does what we want it to do, because msmbuilder.io.saveh
                # deletes keys that exist already. However, this could be made more
                # efficient by only updating the XYZList node since it's the
                # only thing that changes

                new_traj_locs = np.concatenate((old_locs, file_list[old_num_files:old_num_files + n_loaded]))
                self.project._traj_converted_from[old_ind] = list(new_traj_locs)
                self.project._traj_lengths[old_ind] = traj.n_frames
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
            logger.warn('Fewer frames found than currently have. Skipping. (%s)' % traj_loc)

    def convert(self):
        """
        Main method for this class. Convert all of the trajectories into
        lh5 format and save them to self.output_traj_dir.

        Returns
        -------
        project : msmbuilder.project
            The project object, summarizing the conversion
        """

        if not self.project is None:
            old_traj_locs = ['/'.join([d for d in os.path.relpath(file_list[0]).split('/') if not d in ['.', '..']][:-1])
                             for file_list in self.project._traj_converted_from]
            # ^^^ haha, @rmcgibbo, is this acceptable?
            # basically we want to turn something like:
            #   ../../PROJXXXX/RUN0/CLONE195/frame0.xtc
            # into:
            #   PROJXXXX/RUN0/CLONE195
            # and we use this label to see if we've already included the
            # trajectory in the old project
        else:
            old_traj_locs = []
        old_traj_locs = np.array(old_traj_locs)

        num_trajs_added = 0
        if self.project is None:
            num_orig_trajs = 0
        else:
            num_orig_trajs = len(self.project._traj_paths)

        for file_list in self._input_trajs():
            traj_loc = '/'.join([d for d in os.path.relpath(file_list[0]).split('/')
                                if not d in ['.', '..']][:-1])
            num_files = len(file_list)
            if not traj_loc in old_traj_locs:
                self._load_new_files(file_list, traj_loc)
            else:
                old_ind = np.where(old_traj_locs == traj_loc)[0][0]
                self._update_traj(old_ind, file_list, traj_loc)

        if len(self.traj_paths) == 0 and self.project is None:
            os.rmdir(self.output_traj_dir)
            raise RuntimeError('No conversion jobs found!')

        if not self.project is None:
            self.traj_lengths = list(self.project._traj_lengths) + self.traj_lengths
            self.traj_paths = list(self.project._traj_paths) + self.traj_paths
            self.traj_errors = list(self.project._traj_errors) + self.traj_errors
            self.traj_converted_from = list(self.project._traj_converted_from) + self.traj_converted_from
            self.traj_converted_from = [[str(i) for i in l] for l in self.traj_converted_from]

        self.project = Project({'conf_filename': self.conf_filename_final,
                                'traj_lengths': self.traj_lengths,
                                'traj_paths': self.traj_paths,
                                'traj_errors': self.traj_errors,
                                'traj_converted_from': self.traj_converted_from})

        self.updated_project = True

    def _input_trajs(self):
        logger.warning("WARNING: Sorting trajectory files by numerical values in their names.")
        logger.warning("Ensure that numbering is as intended.")

        traj_dirs = glob(os.path.join(self.input_traj_dir, "*"))
        traj_dirs.sort(key=keynat)
        logger.info("Found %s traj dirs", len(traj_dirs))
        for traj_dir in traj_dirs:
            to_add = glob(traj_dir + '/*' + self.input_traj_ext)
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
        traj : mdtraj.Trajectory
        """

        traj = md.load(file_list, discard_overlapping_frames=True,
                       top=self.conf, atom_indices=self.atom_indices)
        # return the number of files loaded, which in this case is all or
        # nothing, since an error is raised if the Trajectory.load_from_<ext>
        # doesn't work
        return traj, len(file_list)


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
            clone_dirs.sort(key=keynat)
            logger.info("%s: Found %d CLONE dirs", run_dir, len(clone_dirs))

            for clone_dir in clone_dirs:
                to_add = glob(clone_dir + '/*' + self.input_traj_ext)
                to_add.sort(key=keynat)
                if to_add:
                    yield to_add

    def _load_traj(self, file_list):
        traj = None

        try:
            traj, n_loaded = super(FahProjectBuilder, self)._load_traj(file_list)
        except (RuntimeError, IOError) as e:

            if hasattr(e, "errno") and e.errno == 2:  # Then the pdb filename doesn't exist
                raise e

            corrupted_files = True
            n_corrupted = 1
            logger.error("Some files appear to be corrupted")
            while corrupted_files:
                logger.error("Trying to recover by discarding the %d-th-to-last file", n_corrupted)
                if len(file_list[:-n_corrupted]) == 0:
                    traj = None
                    break
                try:
                    traj, n_loaded = super(FahProjectBuilder, self)._load_traj(
                        file_list[:-n_corrupted])
                except IOError:
                    n_corrupted += 1
                else:
                    logger.error("That seemed to work")
                    corrupted_files = False

        if traj is None:
            raise RuntimeError("Corrupted frames in %s, recovery impossible" % file_list)

        return traj, n_loaded
