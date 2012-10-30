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

"""Trajectory stores a sequence of conformations.
"""

import copy
import os
import tables
import numpy as np

from msmbuilder import pdb
from msmbuilder import io
from msmbuilder.Conformation import ConformationBaseClass, Conformation
from msmbuilder import xtc
from msmbuilder import dcd

import logging
logger = logging.getLogger(__name__)

MAXINT16 = np.iinfo(np.int16).max
MAXINT32 = np.iinfo(np.int32).max
DEFAULT_PRECISION = 1000


def _convert_to_lossy_integers(data, precision=DEFAULT_PRECISION):
    """Implementation of the lossy compression used in Gromacs XTC using the 
    pytables library.  Convert 32 bit floats into 16 bit integers.  These 
    conversion functions have been optimized for memory use.  Further memory 
    reduction would require an in-place astype() operation, which one could 
    create using ctypes."""
    if (np.max(data)*float(precision) < MAXINT16 and 
        np.min(data)*float(precision) > -MAXINT16):

        data *= float(precision) 
        rounded_data = data.astype("int16")
        data /= float(precision) 
    else:
        data *= float(precision)
        rounded_data = data.astype("int32")
        data /= float(precision)
        logger.error("Data range too large for int16: try removing center of " 
                     "mass motion, check for 'blowing up, or just use .h5 or " 
                     ".xtc format.'")
    return rounded_data


def _convert_from_lossy_integers(data, precision=DEFAULT_PRECISION):
    """Implementation of the lossy compression used in Gromacs XTC using the 
    pytables library.  Convert 16 bit integers into 32 bit floats."""
    data_copy = data.astype("float32")
    data_copy /= float(precision)
    return data_copy


class Trajectory(ConformationBaseClass):
    """This is the representation of a sequence of  conformations.

    Notes:
    Use classmethod load_from_pdb to create an instance of this class from a 
    PDB filename.
    The Trajectory is a dictionary-like object.
    The dictionary key 'XYZList' contains a numpy array of XYZ coordinates, 
    stored such that:
    X[i,j,k] gives Frame i, Atom j, Coordinate k.
    """
    def __init__(self, conf):
        """Create a Trajectory from a single conformation.

        Leaves the XYZList key as an empty list for easy appending.

        Parameters
        ----------
        S: Conformation
            The single conformation to build a trajectory from

        Notes
        -----
        You probably want to use something else to load trajectories

        See Also
        --------
        load_from_pdb
        load_from_hdf
        load_from_lhdf
        """

        ConformationBaseClass.__init__(self, conf)
        self["XYZList"] = []
        if "XYZList" in conf:
            self["XYZList"] = conf["XYZList"].copy()

    def subsample(self, stride):
        """Keep only the frames at some interval

        Shorten trajectory by taking every `stride`th frame. Note that this is
        an inplace operation!

        Parameters
        ----------
        stride: int
            interval

        Notes
        -----
        Trajectory supports fancy indexing, so you can do this by yourself

        Example
        -------
        >> mytrajectory.subsample(10)

        >> mytrajectory[::10]

        These do the same thing except that one is inplace and one is not

        """
        if stride == 1:
            return

        self["XYZList"] = self["XYZList"][::stride].copy()

    def __getitem__(self, key):
        if (isinstance(key, int) or isinstance(key, slice) or 
            isinstance(key, np.ndarray)):

            if isinstance(key, int):
                key = [key]

            newtraj = copy.copy(self)
            newtraj['XYZList'] = self['XYZList'][key]

            return newtraj

        return super(Trajectory, self).__getitem__(key)

    def __len__(self):
        return len(self['XYZList'])

    def __add__(self, other):
        """Addition means concatenation of two trajectory instances"""
        # Check type of other
        if not isinstance(other, Trajectory):
            raise TypeError('You can only add two Trajectory instances')
        concatenated_trajs = copy.deepcopy(self)
        # Simply copy the XYZList in here if the Trajectory is Empty.
        if 'XYZList' not in self and 'XYZList' not in other:
            pass
        elif 'XYZList' not in self:
            concatenated_trajs['XYZList'] = copy.deepcopy(other['XYZList'])
        elif 'XYZList' not in other:
            concatenated_trajs['XYZList'] = copy.deepcopy(self['XYZList'])
        else:
            if self['XYZList'].shape[1] != other['XYZList'].shape[1]:
                raise TypeError("The two trajectories don\'t have the same " 
                                "number of atoms")
            concatenated_trajs['XYZList'] = np.vstack((self['XYZList'], 
                                                       other['XYZList']))

        return concatenated_trajs

    def __iadd__(self, other):
        """Addition means concatenation of two trajectory instances"""
        # NOTE: CRS thinks this should just use __add__ like so:
        return self + other

        ## Check type of other
        #if not isinstance(other, Trajectory):
        #    raise TypeError('You can only add two Trajectory instances')
        ## Simply copy the XYZList in here if the Trajectory is Empty.
        #if 'XYZList' not in self:
        #    self['XYZList'] = copy.deepcopy(other['XYZList'])
        #else:
        #    # Check number of atoms.
        #    if not self['XYZList'].shape[1] == other['XYZList'].shape[1]:
        #        raise TypeError('The two trajectories don\'t have the same ' 
        #                         'number of atoms')
        #    self['XYZList'] = np.vstack((self['XYZList'], other['XYZList']))
        #return self

    def restrict_atom_indices(self, atom_indices):
        ConformationBaseClass.restrict_atom_indices(self, atom_indices)

        self['XYZList'] = copy.copy(self['XYZList'][:, atom_indices])

    def save_to_lhdf(self, filename, precision=DEFAULT_PRECISION):
        """Save a Trajectory instance to a Lossy HDF File.

        First, remove the XYZList key because it should be written using the
        special CArray operation.  This file format is roughly equivalent to
        an XTC and should comparable file sizes but with better IO performance.

        Parameters
        ----------
        filename: str
            location to save to
        precision : float, optional
            Precision to save xyzlist
        """
        # This doesn't get saved it's just regenerated every time the 
        # trajectory is loaded
        index_list = self.pop('IndexList') 
        
        xyzlist = self.pop('XYZList')
        rounded = _convert_to_lossy_integers(xyzlist, precision)
        self['XYZList'] = rounded

        io.saveh(filename, **self)

        # Return the instance to what it was before the save:
        self['XYZList'] = xyzlist
        self['IndexList'] = index_list 
        
    def save_to_xtc(self, filename, precision=DEFAULT_PRECISION):
        """Dump the coordinates to XTC

        Parameters
        ----------
        Filename: str
            location to save to
        precision: float, optional
            I'm not really sure what this does (RTM 6/27).
        """
 
        if os.path.exists(filename):
            raise IOError("%s already exists" % filename)

        xtc_file = xtc.XTCWriter(filename)

        for i in range(len(self["XYZList"])):
            xtc_file.write(self["XYZList"][i], 1, i, \
                            np.eye(3, 3, dtype='float32'), precision)

    def save_to_pdb(self, filename):
        """Dump the coordinates to PDB

        Parameters
        ----------
        Filename: str
            location to save to

        Notes
        -----
        Don't use this for a very big trajectory. 
        PDB is plaintext and takes a lot of memory
        """

        for i in range(len(self["XYZList"])):
            pdb.write_pdb_conformation(filename, self["AtomID"], 
                                       self["AtomNames"], self["ResidueNames"],
                                       self["ResidueID"], self["XYZList"][i],
                                       self["ChainID"])

    def save_to_xyz(self, filename):
        """Dump the coordinates to XYZ format

        Parameters
        ----------
        Filename: str
            location to save to

        Notes
        -----
        TODO: What exactly is the XYZ format? (RTM 6/27) 
            CRS: I think tinker uses it?
        """

        output_lines = []
        title = 'From MSMBuilder save_to_xyz function'
        for i in range(len(self["XYZList"])):
            xyz = self["XYZList"][i]
            num_atoms = xyz.shape[0]
            output_lines.append('%i' % num_atoms)
            output_lines.append(title)
            for j in range(xyz.shape[0]):
                output_lines.append("%-5s% 12.6f% 12.6f% 12.6f\n" % 
                                    (self["AtomNames"][j], xyz[j, 0], 
                                     xyz[j, 1], xyz[j, 2]))
                # The above line seems to me to have an extra % 
                # for no reason: "%-5s%<-here" 

        with open(filename, 'w') as f:
            f.writelines(output_lines)

    def save(self, filename, precision=DEFAULT_PRECISION):
        """Dump the coordinates to disk in format auto-detected by filename

        Parameters
        ----------
        Filename: str
            location to save to

        Notes
        -----
        Formats supported are h5, xtc, pdb, lh5 and xyz
        """

        extension = os.path.splitext(filename)[1]

        if extension == '.h5':
            self.save_to_hdf(filename)
        elif extension == '.xtc':
            self.save_to_xtc(filename)
        elif extension == '.pdb':
            self.save_to_pdb(filename)
        elif extension == '.lh5':
            self.save_to_lhdf(filename, precision=precision)
        elif extension == '.xyz':
            self.save_to_xyz(filename)
        else:
            raise IOError("File: %s. I don't understand the extension '%s'" %  
                            (filename, extension) )

    def append_pdb(self, filename):
        """Add a pdb file to self.

        Parameters
        ----------
        filename: str
            location to find a single conformation to append
    
        """

        self += Trajectory.load_from_pdb(filename)

    @classmethod
    def load_from_pdb(cls, filename):
        """Create a Trajectory from a PDB Filename

        Parameters
        ----------
        filename: str
            location to load from
        """
        return Trajectory(pdb.load_pdb(filename, all_frames=True)) 

    @classmethod
    def load_from_xtc(cls, xtc_filename_list, pdb_filename=None, conf=None,
                      just_inspect=False, discard_overlapping_frames=False):
        """Create a Trajectory from a collection of XTC files

        Parameters
        ----------
        xtc_filename_list: list
            list of files to load from
        pdb_filename: str, optional
            XTC format doesnt have the connectivity information, which needs 
            to be supplied. You can either supply it by giving a path to the 
            PDB file (here) or by supplying a Conf or Traj object containing 
            the right connectivity (next arg)
        conf: Conformation, optional
            A conformation (actually passing another trajectory will work) that
            has the right atom labeling
        just_inspect: bool, optional
            Dont actually load, just return dimensions
        discard_overlapping_frames: bool, optional
            Check for redundant frames and discard them. 
            (RTM 6/27 should this be default True?)

        Returns
        -------
        trajectory: Trajectory
            Trajectory loaded from disk if just_inspect=False

        shape: tuple
            shape of the trajectory if just_inspect=True
        """
        
        # temp_traj is the to be trajectory object, but each key is added in 
        # steps, so it is not complete until the very end
        if pdb_filename != None:
            temp_traj = Trajectory.load_from_pdb(pdb_filename)
        elif conf != None:
            temp_traj = Trajectory(conf)
        else:
            raise Exception("ERROR: Need a conformation to construct a " 
                            "trajectory.")

        if not just_inspect:

            temp_traj["XYZList"] = []
            num_redundant = 0

            for i, c in enumerate(xtc.XTCReader(xtc_filename_list)):
                # check to see if we have redundant frames as we load them up
                if discard_overlapping_frames:
                    if i > 0:
                        if (np.sum(np.abs(c.coords - temp_traj["XYZList"][-1]))
                            < 10.**-8):
                            num_redundant += 1
                    temp_traj["XYZList"].append(np.array(c.coords).copy())

                else:
                    temp_traj["XYZList"].append(np.array(c.coords).copy())

            temp_traj["XYZList"] = np.array(temp_traj["XYZList"])
            if num_redundant != 0:
                logger.warning("Found and discarded %d redundant snapshots in" 
                               " loaded traj", num_redundant)
        else: # just_inspect is True, so return the shape of the data
            i = 0
            for c in xtc.XTCReader(xtc_filename_list):
                if i == 0:
                    conf_shape = np.shape(c.coords)
                i += 1
                # This loop assumes that all of the XTC's are the SAME. This 
                # is true for F@h datasets, so I suppose it's alright. But to 
                # be general we should check every XTC, I think
            shape = np.array((i, conf_shape[0], conf_shape[1]))
            return shape

        return temp_traj

    @classmethod
    def load_from_dcd(cls, filename_list, pdb_filename=None, conf=None, 
                      just_inspect=False):
        """Create a Trajectory from a Filename."""

        # temp_traj is the to be trajectory object, but each key is added in 
        # steps, so it is not complete until the very end

        if pdb_filename != None:
            temp_traj = Trajectory.load_from_pdb(pdb_filename)
        elif conf != None:
            temp_traj = Trajectory(conf)
        else:
            raise Exception("ERROR: Need a conformation to construct a "
                            "trajectory.")

        if not just_inspect:
            temp_traj["XYZList"] = []
            for c in dcd.DCDReader(filename_list):
                temp_traj["XYZList"].append(c.copy())
            temp_traj["XYZList"] = np.array(temp_traj["XYZList"])
        else:  # This is wasteful to read everything in just to get the length
            xyz = []
            for c in dcd.DCDReader(filename_list):
                xyz.append(c.copy())
            xyz = np.array(xyz)
            return xyz.shape

        return temp_traj

    @classmethod
    def load_from_trr(cls, trr_filename_list, pdb_filename=None, conf=None, 
                      just_inspect=False):
        """Load a trajectory from a single or many .trr files."""
        # temp_traj is the to be trajectory object, but each key is added in 
        # steps, so it is not complete until the very end
        if pdb_filename != None:
            temp_traj = Trajectory.load_from_pdb(pdb_filename)
        elif conf != None:
            temp_traj = Trajectory(conf)
        else:
            raise Exception("ERROR: Need a conformation to construct a "
                            "trajectory.")
        if not just_inspect:
            temp_traj["XYZList"] = []
            temp_traj["Velocities"] = [] # Do we care about these? (and Forces)
            temp_traj["Forces"] = []
            for c in xtc.TRRReader(trr_filename_list):
                temp_traj["XYZList"].append(np.array(c.coords).copy())
                temp_traj["Velocities"].append(np.array(c.velocities).copy())
                temp_traj["Forces"].append(np.array(c.forces).copy())
            temp_traj["XYZList"] = np.array(temp_traj["XYZList"])
            temp_traj["Velocities"] = np.array(temp_traj["Velocities"])
            temp_traj["Forces"] = np.array(temp_traj["Forces"])
        else:
            i = 0
            for c in xtc.TRRReader(trr_filename_list):
                if i == 0:
                    conf_shape = np.shape(c.coords)
                i += 1
            shape = np.array((i, conf_shape[0], conf_shape[1]))
            return shape
        return temp_traj

    @classmethod
    def load_from_pdb_list(cls, filenames):
        """Load a trajectory from one or more PDB files"""
        # temp_traj is the to be trajectory object, but each key is added in 
        # steps, so it is not complete until the very end
        temp_traj = Trajectory.load_from_pdb(filenames[0])
        for f in filenames[1:]:
            temp_traj.append_pdb(f) # AHA! This is what uses that function.
        temp_traj["XYZList"] = np.array(temp_traj["XYZList"])
        return temp_traj

    @classmethod
    def enum_chunks_from_hdf(cls, traj_filename, stride=None, 
                             atom_indices=None, chunk_size=50000):
        """
        Function to read trajectory files which have been saved as HDF.

        This function is iterable, so should be used like:

        from msmbuilder import Trajectory
        for trajectory_chunk in Trajectory.enum_chunks_from_hdf(filename):
            ... # Do something with each chunk. 
                # The chunk looks like a regular Trajectory instance

        Inputs:
        - traj_filename: Filename to find the trajectory
        - stride [None]: Integer number of frames to subsample the trajectory
        - atom_indices [None]: np.ndarray of atom indices to read in 
            (0-indexed)
        - chunk_size [100000]: Integer number of frames to read in a chunk
            NOTE: chunk_size will change in order to be a multiple of the input                   stride. This is necessary in order to make sure the Stride 
                  and chunks line up

        Outputs:
        - Nothing. This is an iterable function, so it yields Trajectory 
            instances
        """
        restrict_atoms = False
        if atom_indices != None:
            restrict_atoms = True
        if stride != None:
            while (chunk_size % stride) != 0:# Need to do this in order to make 
                                             # sure we stride correctly.
                                             # since we read in chunks, and 
                                             # then we need the strides
                                             # to line up
                chunk_size -= 1

        temp_traj = {}
        file_obj = tables.File(traj_filename,'r')
        # load all the data other than XYZList

        if restrict_atoms:
            temp_traj['AtomID'] = np.array(file_obj.root.AtomID[atom_indices], 
                                           dtype=np.int32)
            temp_traj['AtomNames'] = np.array(file_obj.\
                                              root.AtomNames[atom_indices])
            temp_traj['ChainID'] = np.array(file_obj.\
                                            root.ChainID[atom_indices])
            temp_traj['ResidueID'] = np.array(file_obj.\
                                              root.ResidueID[atom_indices],
                                              dtype=np.int32)
            temp_traj['ResidueNames'] = np.array(file_obj.root.\
                                                 ResidueNames[atom_indices])
            # The above line continuations look messy, but I don't see any 
            # other way to do it, other than go above 80 chars. What do 
            # people think?

        else:
            temp_traj['AtomID'] = np.array(file_obj.root.AtomID[:], 
                                           dtype=np.int32)
            temp_traj['AtomNames'] = np.array(file_obj.root.AtomNames[:])
            temp_traj['ChainID'] = np.array(file_obj.root.ChainID[:])
            temp_traj['ResidueID'] = np.array(file_obj.root.ResidueID[:], 
                                              dtype=np.int32)
            temp_traj['ResidueNames'] = np.array(file_obj.root.ResidueNames[:])
            
        # IndexList will be generated when the Trajectory object is 
        # instantiated

        # Loaded everything except XYZList

        shape = file_obj.root.XYZList.shape
        begin_range_list = np.arange(0, shape[0], chunk_size)
        end_range_list = np.concatenate((begin_range_list[1:], [shape[0]]))

        for r0, r1 in zip(begin_range_list, end_range_list):
            if restrict_atoms:
                temp_traj['XYZList'] = np.array(file_obj.\
                                                root.XYZList[r0:r1:stride, 
                                                             atom_indices])
            else:
                temp_traj['XYZList'] = np.array(file_obj.\
                                                root.XYZList[r0:r1:stride])

            yield cls(temp_traj)

        file_obj.close()

        return

    @classmethod
    def enum_chunks_from_lhdf(cls, traj_filename, precision=DEFAULT_PRECISION, 
                              stride=None, atom_indices=None, 
                              chunk_size=50000):
        """
        Method to read trajectory files which have been saved as LHDF.
        Note that this method simply calls the enum_chunks_from_hdf method.

        This function is an iterable, so should be used like:

        from msmbuilder import Trajectory
        for trajectory_chunk in Trajectory.enum_chunks_from_lhdf(filename):
            ... # Do something with each chunk. The chunk looks like a 
                # regular Trajectory instance

        Inputs:
        - traj_filename: Filename to find the trajectory
        - precision [1000]: precision used when saving as lossy integers
        - stride [None]: Integer number of frames to subsample the trajectory
        - atom_indices [None]: np.ndarray of atom indices to read in (0-indexed)
        - chunk_size [100000]: Integer number of frames to read in a chunk
            NOTE: ChunkSize will change in order to be a multiple of the input 
                  stride. This is necessary in order to make sure the Stride 
                  and chunks line up

        Outputs:
        - Nothing. This is an iterable function, so it yields Trajectory 
            instances
        """
        for temp_traj in cls.enum_chunks_from_hdf(traj_filename, stride, 
                                                  atom_indices, chunk_size):

            temp_traj['XYZList'] = _convert_from_lossy_integers(
                                                  temp_traj['XYZList'],
                                                  precision)

            yield temp_traj

        return

    @classmethod
    def load_from_hdf(cls, traj_filename, just_inspect=False, stride=None, 
                      atom_indices=None, chunk_size=50000):
        """
        Method to load a trajectory which was saved as HDF

        Inputs:
        - traj_filename: Filename to find the trajectory
        - just_inspect [False]: If True, then the method returns the shape of 
            the XYZList stored on disk
        - stride [None]: Integer number of frames to subsample the trajectory
        - atom_indices [None]: np.ndarray of atom indices to read in (0-indexed)
        - chunk_size [100000]: Integer number of frames to read in a chunk
            NOTE: ChunkSize will change in order to be a multiple of the input 
                  stride. This is necessary in order to make sure the Stride 
                  and chunks line up

        Outputs:
        - trajectory: Trajectory instance read from disk
        """
        if not just_inspect:
            chunk_list = list(cls.enum_chunks_from_hdf(
                                                     traj_filename,
                                                     stride=stride,
                                                     atom_indices=atom_indices,
                                                     chunk_size=chunk_size))
            temp_traj = chunk_list[0]
            temp_traj['XYZList'] = np.concatenate([t['XYZList'] 
                                                   for t in chunk_list])
            return temp_traj

        else:
            file_obj = tables.File(traj_filename)
            shape = file_obj.root.XYZList.shape
            file_obj.close()
            return shape

    @classmethod
    def load_from_lhdf(cls, traj_filename, just_inspect=False, 
                       precision=DEFAULT_PRECISION, stride=None, 
                       atom_indices=None, chunk_size=50000):
        """
        Method to load a trajectory which was saved as LHDF

        Inputs:
        - traj_filename: Filename to find the trajectory
        - just_inspect [False]: If True, then the method returns the shape of 
            the XYZList stored on disk
        - stride [None]: Integer number of frames to subsample the trajectory
        - precision [1000]: precision used when saving as lossy integers
        - AtomIndices [None]: np.ndarray of atom indices to read in (0-indexed)
        - ChunkSize [100000]: Integer number of frames to read in a chunk
            NOTE: ChunkSize will change in order to be a multiple of the input
                  stride. This is necessary in order to make sure the Stride 
                  and chunks line up

        Outputs:
        - trajectory: Trajectory instance read from disk
        """
        if not just_inspect:
            temp_traj = cls.load_from_hdf( traj_filename, stride=stride, 
                                           atom_indices=atom_indices)
            temp_traj['XYZList'] = _convert_from_lossy_integers(
                                           temp_traj['XYZList'], precision)
            return temp_traj

        else:
            file_obj = tables.File(traj_filename)
            shape = file_obj.root.XYZList.shape
            file_obj.close()
            return shape

    @classmethod
    def read_xtc_frame(cls, traj_filename, which_frame):
        """Read a single frame from XTC trajectory file without loading the 
        entire file into memory."""
        i = 0
        for c in xtc.XTCReader(traj_filename):
            if i == which_frame:
                return np.array(c.coords)
            i += 1
        raise Exception("Frame %d not found in file %s; last frame found was "\
                        "%d" % (WhichFrame, TrajFilename, i))

    @classmethod
    def read_dcd_frame(cls, traj_filename, which_frame):
        """Read a single frame from DCD trajectory without loading the entire
        file into memory."""
        reader = dcd.DCDReader(traj_filename, firstframe=which_frame, 
                               lastframe=which_frame)
        xyz = None
        for c in reader:
            xyz = c.copy()
        if xyz == None:
            raise Exception("Frame %s not found in file %s." % (WhichFrame, 
                                                                TrajFilename))
        return xyz

    @classmethod
    def read_hdf_frame(cls, traj_filename, which_frame):
        """Read a single frame from HDF5 trajectory file without loading the 
        entire file into memory."""
        file_obj = tables.File(traj_filename)
        xyz = file_obj.root.XYZList[which_frame]
        file_obj.close()
        return xyz

    @classmethod
    def read_lhdf_frame(cls, traj_filename, which_frame, \
                        precision=DEFAULT_PRECISION):
        """Read a single frame from Lossy LHDF5 trajectory file without loading 
        entire file into memory."""
        xyz = cls.read_hdf_frame(traj_filename, which_frame)
        xyz = _convert_from_lossy_integers(xyz, precision)
        return xyz

    @classmethod
    def read_frame(cls, traj_filename, which_frame, conf=None):
        extension = os.path.splitext(traj_filename)[1]

        if extension == '.xtc':
            return Trajectory.read_xtc_frame(traj_filename, which_frame)
        elif extension == '.h5':
            return Trajectory.read_hdf_frame(traj_filename, which_frame)
        elif extension == '.lh5':
            return Trajectory.read_lhdf_frame(traj_filename, which_frame)
        elif extension == '.dcd':
            return Trajectory.read_dcd_frame(traj_filename, which_frame)
        else:
            raise IOError("Incorrect file type--cannot get conformation %s" % 
                           traj_filename)

    @classmethod
    def load_trajectory_file(cls, filename, just_inspect=False, conf=None, 
                             stride=1, atom_indices=None):
        """Loads a trajectory into memory, automatically deciding which methods
        to call based on filetype.  For XTC files, this method uses a 
        pre-registered Conformation filename as a pdb."""
        
        extension = os.path.splitext(filename)[1]
        
        # check to see if we're supposed to load only a subset of the atoms
        if atom_indices != None:
            if (extension == '.lh5') or (extension == '.h5'):
                pass # we deal with this below
            else:
                raise NotImplementedError('AtomIndices kwarg option only'
                                          'available for .lh5 & .h5 format')
            
        # if we're not going to load a subset of the atoms, then proceed 

        elif extension == '.xtc':
            if conf == None:
                raise Exception("Need to register a Conformation to use XTC "
                                "Reader.")
            return Trajectory.load_from_xtc(filename, conf=conf, 
                                           just_inspect=just_inspect)[::stride]

        elif extension == '.dcd':
            if conf == None:
                raise Exception("Need to register a Conformation to use DCD "
                                "Reader.")
            return Trajectory.load_from_dcd(filename, conf=conf, 
                                           just_inspect=just_inspect)[::stride]

        if extension == '.h5':
            return Trajectory.load_from_hdf(filename, 
                                            just_inspect=just_inspect,
                                            stride=stride, 
                                            atom_indices=atom_indices)

        elif extension == '.lh5':
            return Trajectory.load_from_lhdf(filename, 
                                             just_inspect=just_inspect, 
                                             stride=stride, 
                                             atom_indices=atom_indices)

        elif extension == '.pdb':
            return Trajectory.load_from_pdb(filename)[::stride]

        else:
            raise IOError("File: %s. I don't understand the extension '%s'" % 
                          (filename, extension))

    @classmethod
    def append_frames_to_file(cls, filename, xyz_list, 
                              precision=DEFAULT_PRECISION, 
                              discard_overlapping_frames=False):
        """Append an array of XYZ data to an existing .h5 or .lh5 file.
        """

        extension = os.path.splitext(filename)[1]

        if extension in [".h5", ".lh5"]:
            file_obj = tables.File(filename, "a")
        else:
            raise Exception("File must be .h5 or .lh5")

        if not file_obj.root.XYZList.shape[1:] == xyz_list.shape[1:]:
            raise Exception("Error: data cannot be appended to trajectory " 
                            "due to incorrect shape.")

        if extension == ".h5":
            to_add = xyz_list
        elif extension == ".lh5":
            to_add = _convert_to_lossy_integers(xyz_list, precision)

        # right now this only checks for the last written frame
        if discard_overlapping_frames:
            while (file_obj.root.XYZList[-1, :, :] == to_add[0, :, :]).all():
                to_add = to_add[1:, :, :]

        file_obj.root.XYZList.append(to_add)

        file_obj.flush()
        file_obj.close()

    @classmethod
    def _reduce_redundant_snapshots(cls, trajectory):
        """ Takes a trajectory object, and removes data from the 'XYZList' 
        entry that are duplicated snapshots. Does this by checking if two 
        contiguous snapshots are binary equivalent.
        """

        not_done = True
        i = 0  # index for the snapshot we're working on
        n = 0  # counts the number of corrections

        while not_done:

            # check to see if we are done
            if i == trajectory["XYZList"].shape[0] - 1:
                break

            if (trajectory["XYZList"][i, :, :] == 
                trajectory["XYZList"][i+1, :, :]).all():

                trajectory = trajectory[:i] + trajectory[i+1:]
                n += 1
            else:
                i += 1

        if n != 0:
            logger.warning("Found and eliminated %d redundant snapshots " 
                           "in trajectory", n)

        return trajectory
