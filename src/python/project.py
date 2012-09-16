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
from msmbuilder import msmio
import logging
from glob import glob
from msmbuilder.utils import keynat
logger = logging.getLogger('project')

class Project(object):
    
    # this maps the private instance variable names to
    # the names that they can be serialized under
    mapping = {
        # the default name comes first, and alternate names
        # can come after
        '_conf_filename': ['conf_filename', 'ConfFilename'],
        '_n_trajs': ['n_trajs', 'NumTrajs'],
        '_traj_lengths': ['traj_lengths', 'TrajLengths'],
        '_traj_basename': ['traj_basename', 'TrajFileBaseName'],
        '_traj_path': ['traj_path', 'TrajFilePath'],
        '_traj_ext': ['traj_ext', 'TrajFileType']
    }
    
    @property
    def conf_filename(self):
        """Filename of the project topology (PDB)"""
        return os.path.normpath(os.path.join(self._project_dir, self._conf_filename))
    @property
    def n_trajs(self):
        """Number of trajectories in the project"""
        return self._n_trajs
    @property
    def traj_lengths(self):
        """Length of each of the trajectories, in frames"""
        return self._traj_lengths
    
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
            
        See Also
        --------
        load_from : to load from disk
        """
        
        self._conf_filename = None
        self._n_trajs = None
        self._traj_lengths = None
        self._traj_basename = None
        self._traj_ext = None
        self._conf_filename = None
        self._project_dir = os.path.abspath(project_dir)
                
        records_keys = set(records.keys())
        for key, val in self.mapping.iteritems():
            match = list(set(val) & records_keys)
            if match:
                assert len(match) == 1, 'duplicate keys detected'
                setattr(self, key, records[match[0]])
        
        for key in self.mapping.keys():
            if getattr(self, key) is None:
                raise ValueError("attribute missing: %s" % key)
        
        self._traj_lengths = np.array(self._traj_lengths)
        
        if validate:
            self._validate()
        
    def __repr__(self):
        "Return a string representation of the project"
        return 'Project<(%d trajectories, from %s to %s  topology: %s)>' % (self.n_trajs, self.traj_filename(0), self.traj_filename(self.n_trajs-1), self.conf_filename)
    
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
                records = yaml.load(f)
                    
        elif filename.endswith('.h5'):
            records = msmio.loadh(filename, deferred=False)
                
            # UGLY: everything comes out of the .h5 file as an array
            # even the stuff like ConfFilename that is really just
            # a string
            for key, value in records.iteritems():
                if key in cls.mapping['_traj_lengths']:
                    records[key] = value.tolist()
                else:
                    records[key] = np.asscalar(value)
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
            
        records = {}
        for key, val in self.mapping.iteritems():
            records[val[0]] = sanitize_type(getattr(self, key))
        
        # somewhat complicated logic if the directory you're
        # saving in is different than the directory this
        # project references its paths from
        
        # the point is that the when the file lists paths, those
        # paths are going to be interpreted as being with respect to
        # the directory that the file is in. So when the Project file
        # is being resaved (but the Trajectorys are not being moved)
        # then the paths need to change to compensate
        relative = os.path.relpath(self._project_dir, os.path.dirname(filename_or_file))
        records['traj_path'] = os.path.join(relative, records['traj_path'])
        records['conf_filename'] = os.path.join(relative, records['conf_filename'])

        yaml.dump(records, handle)
        
        if own_fid:
            handle.close()
            
        return filename_or_file

    def load_traj(self, trj_index, stride=1):
        filename = self.traj_filename(trj_index)
        return Trajectory.LoadTrajectoryFile(filename, Stride=stride)
            
    def traj_filename(self, traj_index):
        return os.path.normpath(os.path.join(self._project_dir, self._traj_path,
                self._traj_basename + str(traj_index) + self._traj_ext))
        
    def _validate(self):
        if not self._n_trajs == len(self.traj_lengths):
            raise ValueError('traj lengths mismatch')
        if not os.path.exists(self.conf_filename):
            raise ValueError('conf does not exist: %s' % self.conf_filename)
        for i in xrange(self.n_trajs):
            if not os.path.exists(self.traj_filename(i)):
                raise ValueError("%s does not exist" % self.traj_filename(i))
        if not np.all(self.traj_lengths == self._eval_traj_lengths()):
            raise ValueError('Trajs length don\'t match what\'s on disk')

    
    def empty_trajectory(self):
        traj = Trajectory.LoadTrajectoryFile(self.conf_filename)
        traj['XYZList'] = None
        return traj
        
    def _eval_traj_lengths(self, as_numpy=False):
        traj_lengths = np.zeros(self.n_trajs)
        for i in xrange(self.n_trajs):
            traj_lengths[i] = Trajectory.LoadTrajectoryFile(self.traj_filename(i), JustInspect=True, Conf=self.conf_filename)[0]
        return traj_lengths



class ProjectBuilder(object):
    def __init__(self, input_traj_dir, input_traj_ext, conf_filename, **kwargs):
        self.input_traj_dir = input_traj_dir
        self.input_traj_ext = input_traj_ext
        self.conf_filename = conf_filename
        
        self.output_traj_ext = '.lh5'
        self.output_traj_basename = kwargs.pop('output_traj_basename', 'trj')
        self.output_traj_dir = kwargs.pop('output_traj_dir', 'Trajectories')
        self.stride = kwargs.pop('stride', 1)
        
        if len(kwargs) > 0:
            raise ValueError('Unsupported arguments %s' % kwargs.keys())
        if input_traj_ext not in ['.xtc', '.dcd']:
            raise ValueError("Unsupported format")

        self._check_out_dir()
        self.project = self._convert()
        
    def get_project(self):
        return self.project
    
    def _check_out_dir(self):
        if not os.path.exists(self.output_traj_dir):
            os.makedirs(self.output_traj_dir)
        else:
            raise IOError('%s already exists' % self.output_traj_dir)

    def _convert(self):
        traj_lengths = []
        for i, file_list in enumerate(self._input_trajs()):
            traj = self._load_traj(file_list)
            traj["XYZList"] = traj["XYZList"][::self.stride]
            lh5_fn = os.path.join(self.output_traj_dir, (self.output_traj_basename + str(i) + self.output_traj_ext))
            traj.Save(lh5_fn)
            traj_lengths.append(len(traj["XYZList"]))
            
            logger.info("%s, length %d, converted to %s", file_list, traj_lengths[-1], lh5_fn)


        n_trajs = len(traj_lengths)
        
        if n_trajs == 0:
            raise RuntimeError('No conversion jobs found!')

        return Project({'conf_filename': self.conf_filename,
                        'n_trajs': n_trajs,
                        'traj_lengths': np.array(traj_lengths),
                        'traj_basename': self.output_traj_basename,
                        'traj_path': self.output_traj_dir,
                        'traj_ext': self.output_traj_ext})
    
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
        if self.input_traj_ext == '.xtc':
            traj = Trajectory.LoadFromXTC(file_list, PDBFilename=self.conf_filename)
        elif self.input_traj_ext == '.dcd':
            traj = Trajectory.LoadFromXTC(file_list, PDBFilename=self.conf_filename)
        else:
            raise ValueError()
        return traj



def sanitize_type(obj):
    clean_types = [str, int, float, list, dict]
    if any(isinstance(obj, e) for e in clean_types):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise Exception('Cannot sanitize %s' % obj)
    
            