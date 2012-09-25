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
from msmbuilder import Trajectory
from msmbuilder import io
import logging
from msmbuilder.utils import keynat
logger = logging.getLogger('project')


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

    def __init__(self, records, validate=True, project_dir='.'):
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
        This method is generally internally. To load projects from disk,
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
        self._traj_paths = np.array(records['traj_paths'])
        self._traj_converted_from = np.array(records['traj_converted_from'])
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
                ondisk = yaml.load(f)
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
                       'traj_converted_from': [None] * n_trajs,
                       'traj_errors': [None] * n_trajs}

            for i in xrange(n_trajs):
                # this is the convention used in the hdf project format to get the traj paths
                path = os.path.join( ondisk['TrajFilePath'][0], ondisk['TrajFileBaseName'][0] + str(i) + ondisk['TrajFileType'][0] )
                records['traj_paths'].append(path)

        else:
            raise ValueError('Sorry, I can only open files in .yaml'
                             ' or .h5 format: %s' % filename)

        return cls(records, validate=True, project_dir=rootdir)

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
                                    'converted_from': self._traj_converted_from[i].tolist(),
                                    'length': int(self._traj_lengths[i]),
                                    'errors': self._traj_errors[i]})

        yaml.dump(records, handle)

        if own_fid:
            handle.close()

        return filename_or_file

    def load_traj(self, trj_index, stride=1):
        "Load the a trajectory from disk"
        filename = self.traj_filename(trj_index)
        return Trajectory.load_trajectory_file(filename, Stride=stride)

    def load_conf(self):
        "Load the PDB associated with this project from disk"
        return Trajectory.load_trajectory_file(self.conf_filename)

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
        traj['XYZList'] = None
        return traj

    def _eval_traj_shapes(self):
        lengths = np.zeros(self.n_trajs)
        n_atoms = np.zeros(self.n_trajs)
        conf = self.load_conf()
        for i in xrange(self.n_trajs):
            shape = Trajectory.load_trajectory_file(self.traj_filename(i), JustInspect=True, Conf=conf)
            lengths[i] = shape[0]
            n_atoms[i] = shape[1]
        return lengths, n_atoms
