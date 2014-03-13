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

import os
import numpy as np
import yaml
# if CLoader/CDumper are available (i.e. user has libyaml installed)
#  then use them since they are much faster.
try:
    from yaml import CLoader as Loader
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Loader
    from yaml import Dumper

from msmbuilder import MSMLib
import mdtraj as md
from mdtraj import io
import logging
logger = logging.getLogger(__name__)

class Project(object):
    @property
    def conf_filename(self):
        """Filename of the project topology (PDB)"""
        return os.path.normpath(os.path.join(self._project_dir, self._conf_filename))

    @property
    def n_trajs(self):
        """Number of trajectories in the project"""
        return len(self._valid_traj_indices)

    @property
    def traj_lengths(self):
        """Length of each of the trajectories, in frames"""
        return self._traj_lengths[self._valid_traj_indices]
        
    def __eq__(self, other):
        '''Is this project equal to another'''
        if not isinstance(other, Project):
            return False
        return os.path.basename(self._conf_filename) == os.path.basename(other._conf_filename) and \
            np.all(self._traj_lengths == other._traj_lengths) and \
            np.all(np.array([os.path.basename(e) for e in self._traj_paths])
                == np.array([os.path.basename(e) for e in other._traj_paths])) and \
            np.all(self._traj_errors == other._traj_errors)
            # np.all(self._traj_converted_from == other._traj_converted_from)
                                
    def __init__(self, records, validate=False, project_dir='.'):
        """Create a project from a  set of records

        Parameters
        ----------
        records : dict
            The data, either constructed or loaded from disk. If you
            provide insufficient data, we an error will be thrown.
        validate : bool, optional
            If true, some checks for consistency are done
        project_dir : string
            Base directory for the project. Filenames in the records
            dict are assumed to be given relative to this directory

        Notes
        -----
        This method is generally used internally. To load projects from disk,
        use `Project.load_from`

        `records` should be a dict with 'conf_filename`, 'traj_lengths`,
        `traj_paths`, `traj_converted_from` and `traj_errors` keys. The first
        should be a path on disk, relative to `project_dir` giving the pdb for
        the project. the others should be arrays or lists with one entry per
        trajectory in the project.

        See Also
        --------
        load_from : to load from disk
        """

        self._conf_filename = records['conf_filename']
        self._traj_lengths = np.array(records['traj_lengths'])
        self._traj_paths = np.array([os.path.relpath(p) for p in records['traj_paths']])
        self._traj_converted_from = list(records['traj_converted_from'])
        self._traj_errors = np.array(records['traj_errors'])
        self._project_dir = os.path.abspath(project_dir)

        # make sure lengths are all consistent
        if not len(self._traj_lengths) == len(self._traj_paths) == \
                len(self._traj_converted_from) == len(self._traj_errors):
            raise ValueError("Inconsistent records: traj fields not all the same length")

        # set the length field
        self._valid_traj_indices = np.arange(len(self._traj_lengths))

        # if there are errors
        if np.any(self._traj_errors):
            self._valid_traj_indices = np.array([i for i, e in enumerate(self._traj_errors) if e is None], dtype=int)
            n_errors = len([e for e in self._traj_errors if e is not None])
            logger.error('Errors detected in conversion: %d trajectories', n_errors)

            errors = np.setdiff1d(np.arange(len(self._traj_errors)), self._valid_traj_indices)
            logger.error('The trajectories in error are numbers %s. If this project was '
                         'loaded from a yaml file, check the file for details' % errors)

        # now _valid_traj_indicies gives the indices of the trajectories
        # without errors, so that when someone asks for self.traj_filename(i)
        # they'll get self._traj_paths[self._valid_traj_indices[i]]

        if validate:
            self._validate()

    def __repr__(self):
        "Return a string representation of the project"
        return 'Project<(%d trajectories, from %s to %s  topology: %s)>' % (self.n_trajs, self.traj_filename(0), self.traj_filename(self.n_trajs - 1), self.conf_filename)

    @classmethod
    def load_from(cls, filename):
        """
        Load project from disk

        Parameters
        ----------
        filename : string
            filename_or_file can be a path to a legacy .h5 or current
            .yaml file.

        Returns
        -------
        project : the loaded project object

        """

        rootdir = os.path.abspath(os.path.dirname(filename))

        if filename.endswith('.yaml'):
            with open(filename) as f:
                ondisk = yaml.load(f, Loader=Loader)
                records = {'conf_filename': ondisk['conf_filename'],
                           'traj_lengths': [],
                           'traj_paths': [],
                           'traj_converted_from': [],
                           'traj_errors': []}

                for trj in ondisk['trajs']:
                    records['traj_lengths'].append(trj['length'])
                    records['traj_paths'].append(trj['path'])
                    records['traj_errors'].append(trj['errors'])
                    records['traj_converted_from'].append(trj['converted_from'])

        elif filename.endswith('.h5'):
            ondisk = io.loadh(filename, deferred=False)
            n_trajs = len(ondisk['TrajLengths'])
            records = {'conf_filename': str(ondisk['ConfFilename'][0]),
                       'traj_lengths': ondisk['TrajLengths'],
                       'traj_paths': [],
                       'traj_converted_from': [ [None] ] * n_trajs,
                       'traj_errors': [None] * n_trajs}

            for i in xrange(n_trajs):
                # this is the convention used in the hdf project format to get the traj paths
                path = os.path.join( ondisk['TrajFilePath'][0], ondisk['TrajFileBaseName'][0] + str(i) + ondisk['TrajFileType'][0] )
                records['traj_paths'].append(path)

        else:
            raise ValueError('Sorry, I can only open files in .yaml'
                             ' or .h5 format: %s' % filename)

        return cls(records, validate=False, project_dir=rootdir)

    def save(self, filename_or_file):
        if isinstance(filename_or_file, basestring):
            if not filename_or_file.endswith('.yaml'):
                filename_or_file += '.yaml'
            dirname = os.path.abspath(os.path.dirname(filename_or_file))
            if not os.path.exists(dirname):
                logger.info("Creating directory: %s" % dirname)
                os.makedirs(dirname)
            handle = open(filename_or_file, 'w')
            own_fid = True
        elif isinstance(filename_or_file, file):
            dirname = os.path.abspath(os.path.dirname(filename_or_file.name))
            handle = filename_or_file
            own_fid = False

        # somewhat complicated logic if the directory you're
        # saving in is different than the directory this
        # project references its paths from

        # the point is that the when the file lists paths, those
        # paths are going to be interpreted as being with respect to
        # the directory that the file is in. So when the Project file
        # is being resaved (but the Trajectorys are not being moved)
        # then the paths need to change to compensate

        relative = os.path.relpath(self._project_dir, os.path.dirname(filename_or_file))

        records = {'trajs': []}
        records['conf_filename'] = os.path.join(relative, self._conf_filename)
        traj_paths = [os.path.join(relative, path) for path in self._traj_paths]
        for i in xrange(len(traj_paths)):
            # yaml doesn't like numpy types, so we have to sanitize them
            records['trajs'].append({'id': i,
                                    'path': str(traj_paths[i]),
                                    'converted_from': list(self._traj_converted_from[i]),
                                    'length': int(self._traj_lengths[i]),
                                    'errors': self._traj_errors[i]})

        yaml.dump(records, handle, Dumper=Dumper)

        if own_fid:
            handle.close()

        return filename_or_file

    def get_random_confs_from_states(self, assignments, states, num_confs, 
        replacement=True, random=np.random):
        """
        Get random conformations from a particular state (or states) in assignments.

        Parameters
        ----------
        assignments : np.ndarray
            2D array storing the assignments for a particular MSM
        states : int or 1d array_like
            state index (or indices) to load random conformations from
        num_confs : int or 1d array_like
            number of conformations to get from state. The shape should 
            be the same as the states argument
        replacement : bool, optional
            whether to sample with replacement or not (default: True)
        random : np.random.RandomState, optional
            use a particular RandomState for generating the random samples.
            this is only useful if you want to get the same samples, i.e.
            when debugging something.

        Returns
        -------
        random_confs : msmbuilder.Trajectory or list of
                       msmbuilder.Trajectory objects
            If states is a list, then the output is a list, otherwise a 
            single trajectory is returned
            Trajectory object containing random conformations from the 
            specified state
        """

        def randomize(state_counts, size=1, replacement=True, random=np.random):
            """
            This is a helper function for selecting random conformations. It will
            select many samples from a discrete, uniform distribution over:
            
            .. math::  \{i\}_{i=1}^{\textnormal{state_counts}}

            If replacement==True, then random.randint will be used, otherwise
            random.permutation will be used.

            Parameters
            ----------
            state_counts : int
                number of conformations in the state
            size : int, optional
                number of samples to draw (size kwarg in np.random.randint)
                default: 1
            replacement : bool, optional
                if True, then we sample with replacement, otherwise we use a 
                permutation
            random : np.random.RandomState, optional
                if you want this to behave deterministically then pass a particular
                random state, otherwise we will use np.random.

            Returns
            -------
            result : np.ndarray
                1d array with samples from the given distribution

            Raises
            ------
            ValueError: if size > state_counts and replacement is False, then
                it is not possible to sample that many conformations without 
                replacement
            """
            assert state_counts > 0

            if replacement:
                result = random.randint(0, state_counts, size=size)
            else:
                if size > state_counts:
                    raise ValueError("Asked for %d conformations from a state "
                                     "with only %d conformations." % (size, state_counts))
                else:
                    result = random.permutation(np.arange(state_counts))[:size]

            return result
            

        if isinstance(states, int):
            states = np.array([states])
        states = np.array(states).flatten()

        # if num_confs is just a number, map it to
        # each state given in states
        if isinstance(num_confs, int):
            num_confs = np.array([num_confs] * len(states)) 
        num_confs = np.array(num_confs).flatten()

        # if num_confs is length-1, then map that value to each
        # state in states
        if len(num_confs) == 1:
            num_confs = np.array(list(num_confs) * len(states))

        if len(num_confs) != len(states):
            raise Exception("num_confs must be the same size as num_states")

        inv_assignments = MSMLib.invert_assignments(assignments)
        state_counts = np.bincount(assignments[np.where(assignments!=-1)])

        random_confs = []

        for n, state in zip(num_confs, states):
            logger.debug("Working on %s", state)
            if state_counts[state] == 0:
                raise ValueError('No conformations to sample from state %d! It contains no assigned conformations.' % state)

            random_conf_inds = randomize(state_counts[state], size=n,
                                         replacement=replacement, 
                                         random=random)

            traj_inds, frame_inds = inv_assignments[state]
            random_confs.append(self.load_frame(traj_inds[random_conf_inds], 
                                                frame_inds[random_conf_inds]))
        
        return random_confs

    def load_traj(self, trj_index, stride=1, atom_indices=None):
        "Load the a trajectory from disk"
        filename = self.traj_filename(trj_index)
        return md.load(filename, stride=stride, atom_indices=atom_indices, discard_overlapping_frames=True)

    def load_chunked_traj(self, trj_index, chunk_size=50000, stride=1, atom_indices=None):
        return md.iterload(self.traj_filename(trj_index), chunk=chunk_size, stride=stride, atom_indices=atom_indices)
        # does anything use this function?

    def load_frame(self, traj_index, frame_index):
        """Load one or more specified frames.

        Parameters
        ----------
        traj_index : int, [int]
            Index or indices of the trajectories to pull from
        frame_index : int, [int]
            Index or indices of the frames to pull from

        Returns
        -------
        traj : msmbuilder.Trajectory
            A trajectory object containing the requested frame(s).
        """

        if np.isscalar(traj_index):
            traj_index = np.array([traj_index])
        if np.isscalar(frame_index):
            frame_index = np.array([frame_index])

        traj_index = np.array(traj_index)
        frame_index = np.array(frame_index)

        if not (traj_index.ndim == 1 and np.all(traj_index.shape == frame_index.shape)):
            raise ValueError('traj_index and frame_index must be 1D and have the same length')

        conf = self.load_conf()
        xyzlist = []
        for i,j in zip(traj_index, frame_index):
            if j >= self.traj_lengths[i]:
                raise ValueError('traj %d too short (%d) to contain a frame %d' % (i, self.traj_lengths[i], j))
                
            xyzlist.append(md.load_frame(self.traj_filename(i), j).xyz)

        conf.xyz = np.concatenate(xyzlist)
        conf.time = [1 for _ in xyzlist]
        conf.unitcell_vectors = None

        return conf

    def load_conf(self):
        "Load the PDB associated with this project from disk"
        return md.load(self.conf_filename)

    def traj_filename(self, traj_index):
        "Get the filename of one of the trajs on disk"
        path = self._traj_paths[self._valid_traj_indices[traj_index]]
        return os.path.normpath(os.path.join(self._project_dir, path))

    def _validate(self):
        "Run some checks to ensure that this project is consistent"
        
        if not os.path.exists(self.conf_filename):
            raise ValueError('conf does not exist: %s' % self.conf_filename)
        for i in xrange(self.n_trajs):
            if not os.path.exists(self.traj_filename(i)):
                raise ValueError("%s does not exist" % self.traj_filename(i))
        lengths, atoms = self._eval_traj_shapes()
        if not np.all(self.traj_lengths == lengths):
            raise ValueError('Trajs length don\'t match what\'s on disk')

        # make sure all trajs have the same number of atoms
        # note that it is possible that there are no valid trajectories, so atoms
        # could be empty
        if len(atoms) > 0 and not np.all(atoms == atoms[0]):
            raise ValueError('Not all trajs have the same number of atoms')

    def empty_traj(self):
        traj = self.load_conf()
        traj.xyz = np.empty((0, traj.n_atoms, 3))
        return traj

    def _eval_traj_shapes(self):
        lengths = np.zeros(self.n_trajs)
        n_atoms = np.zeros(self.n_trajs)
        for i in xrange(self.n_trajs):
            filename = self.traj_filename(i)
            with md.open(filename) as f:
                lengths[i] = len(f)
            n_atoms[i] = md.load_frame(filename, 0).n_atoms
        return lengths, n_atoms
