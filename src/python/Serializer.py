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

"""
The Serializer is a base class for storing dictionary-like objects
to disk as HDF5 files.

Notes:
The Conformation, Trajectory, and Project all inherit Serializer
for HDF5 IO.
"""

import tables
import os
import scipy.sparse
import numpy as np
import warnings

try:
    FILTER = tables.Filters(complevel=9, complib='blosc', shuffle=True)
except:
    warnings.warn("missing BLOSC, no compression will used.", ImportWarning)
    FILTER = tables.Filters()

class Serializer(dict):
    """
    A generic class for dumping dictionaries of data onto disk
    using the pytables HDF5 library.
    
    """
    
    def __init__(self, dict_like_object=None):
        """Build Serializer from dictionary
        
        All Serializer subclass constructors take a dictionary-like object
        as input.  Subclasses will attempt to load (key,value) pairs by name,
        which will raise an exception if something is missing.
        
        Parameters
        ----------
        DictLikeObject : dict, optional
            Data to populate the Serializer with
        
        """
        super(Serializer, self).__init__()
        if dict_like_object is None:
            dict_like_object = {}
        
        self.update(dict_like_object)
    
    def save_to_hdf(self, filename, loc="/", do_file_check=True):
        """Save the contents of the serializer to disk
        
        A generic function for saving ForceFields / Topologies / Conformations
        to H5 files.  Certain types of data cannot be stored as simple arrays,
        so these are the exceptions (if statements) in this function.
        
        Parameters
        ----------
        filename : str
            Filename to save data to
        loc : str, optional
            Resource root in HDF file
        do_file_check : bool, optional
            Check that the location is free before saving
        
        Raises
        ------
        IOError
            If `do_file_check` is True and something is already stored in `Filename`
            
        """
        
        # check h5 file doesn't already exist
        if do_file_check:
            self.check_if_file_exists(filename)
        
        handle = tables.File(filename, 'a')
        
        for key, data in self.iteritems():
            # This checks if the list is homogenous and can be stored
            # in an array data type (ie a square tensor).
            # If not, we need VLArray
            try:
                tmp = np.array(data)
                # This check is necessary for Numpy 1.6.0 and greater,
                # which allow inhomogeneous lists to be converted to dtype
                # arrays.                
                if tmp.dtype == np.dtype("object"):
                    raise ValueError
            except ValueError:
                handle.createVLArray(loc, key, tables.Int16Atom())
                for element in data:
                    handle.getNode(loc, key).append(element)
                continue
            self.save_e_array(np.array(data), key, handle=handle, loc=loc)
        
        handle.flush()
        handle.close()
    
    @classmethod
    def load_from_hdf(cls, filename, loc="/"):
        """Generic function to load HDF files to dict-like object
        
        This is a generic function for loading HDF files into dictionary-like
        objects.  For each subclass, the constructor calls this function, which
        loads all available data.  Then, the class-specific constructor makes a
        few finishing touches.  The different if statements in this function
        refer to exceptions in the way we load things from HDF5.  For example,
        residues cannot be represented as simple arrays, so we have to dance a
        bit to load them.  Similarly, IndexLists (a list of which atom indices
        belong to which residue number) cannot be stored as a simple array, so
        we store them as VLArrays (VL= Variable Length).
        
        Parameters
        ----------
        Filename : str
            Path to resource on disk to load from
        loc : str, option
            Resource root in HDF file
        
        """
        
        serializer = Serializer()
        handle = tables.File(filename, 'r')
        
        for node in handle.listNodes(loc):
            if type(node) == tables.VLArray:
                serializer.update([[node.name, node.read()]])
                continue
            if node[:].size == 1:
                serializer[node.name] = node[:].item()
                continue
            if node.shape[0] == 1:
                serializer[node.name] = np.array(node.read())
                continue
            serializer[node.name] = node[:]
        
        handle.close()
        serializer['SerializerFilename'] = os.path.abspath(filename)
        
        return(cls(serializer))
    
    @staticmethod
    def save_e_array(data, key, filename=None, handle=None, loc="/"):
        """Save dict entry as compressed EArray
        
        E means extensible, which can be useful for extending trajectories.
        
        Parameters
        ----------
        data : np.ndarray
            An array you want to save to disk as hdf5
        key : str
            Key for the array in the HDF5 file
        filename : str
            Filename to sve the data to
        handle : tables.File
            An already opened tables file handle to save the data
            to. One of filename or f0 must be provided
        loc : str
            Resource root in the hdf5 file
        """
        
        if handle is None and filename is None:
            raise ValueError("Must Specify either f0 or filename")
        if handle is None:
            handle = tables.File(filename, 'a')
            close_on_return = True
        else:
            close_on_return = False
            
        if np.rank(data) == 0:
            data = np.array([data])
            
        # the dimension with shape 0 is the one that can be extended
        # along by tables
        shape = np.array(np.shape(data))
        true_shape_0 = shape[0]
        shape[0] = 0
        
        handle.createEArray(loc, key, tables.Atom.from_dtype(data.dtype),
                            shape, filters=FILTER, expectedrows=true_shape_0)
                       
        node = handle.getNode(loc, key)
        node.append(data)

        if close_on_return: 
            handle.close()
    
    @staticmethod
    def save_c_array(data, key, filename=None, handle=None, loc="/"):
        """Save this dictionary entry as a compressed CArray
        
        Parameters
        ----------
        data : np.ndarray
            An array you want to save to disk as hdf5
        key : str
            Key for the array in the HDF5 file
        filename : str
            Filename to sve the data to
        f0 : tables.File
            An already opened tables file handle to save the data
            to. One of filename or f0 must be provided
        loc : str
            Resource root in the hdf5 file
        
        Notes
        -----
        CArray tends to give about 20% better performance than EArray.
        Also, for VHP (576 atoms), Chunkshape[0]=8 seems to give perhaps another
        20%.  The total enhancement appears to be an total of 8800 conformations
        per second versus 6300 for EArray without chunkshape optimization.
        """
        
        if handle is None and filename is None:
            raise ValueError("Must Specify either f0 or filename")
        if handle is None:
            handle = tables.File(filename, 'a')
            close_on_return = True
        else:
            close_on_return = False
        
        if np.rank(data)==0:
            data = np.array([data])
        
        handle.createCArray(loc, key, tables.Atom.from_dtype(data.dtype),
                      np.shape(data), filters=FILTER)
        node = handle.getNode(loc, key)
        node[:] = data
        
        if close_on_return:
            handle.close()
    
    @staticmethod
    def save_csr_matrix(filename, matrix):
        """Save a CSR sparse matrix to disk
        
        Parameters
        ----------
        filename : str
            Location to save to
        matrix : csr_matrix
            matrix to save
        
        Raises
        ------
        TypeError
            If `T` is not a CSR sparse matrix
        IOError
            If a file exists at `Filename`
        """
        
        if not scipy.sparse.isspmatrix_csr(matrix):
            raise TypeError("A CSR sparse matrix is required")

        Serializer({"data" : matrix.data,
                    "indices" : matrix.indices,
                    "indptr" : matrix.indptr,
                    "Shape" : matrix.shape}
                    ).save_to_hdf(filename)

    
    @staticmethod
    def load_csr_matrix(filename):
        """Load a CSR sparse matrix from disk
        
        Parameters
        ----------
        Filename : str
            Path to resource to load from
            
        Returns
        -------
        T : csr_sparse_matrix
            The matrix loaded from disk
        """
        tmp = Serializer.load_from_hdf(filename)
        
        return scipy.sparse.csr_matrix((tmp["data"], tmp["indices"],
            tmp["indptr"]), tmp["Shape"])
    
    @staticmethod
    def save_data(filename, data):
        """Quickly dump an array to disk in h5 format
        
        Writes the array in the 'Data' field
        
        Parameters
        ----------
        Filename : str
            Path to resource to write to
        Data : ndarray
            numpy array to write to disk
        """
        
        Serializer({"Data": data}).save_to_hdf(filename)
    
    @staticmethod
    def load_data(filename):
        """Quickly read an array from disk in h5 format
        
        Read from the 'Data' field of the h5 file
        
        Parameters
        ----------
        Filename : str
            Path to resource to read from
        
        Raises 
        ------
        KeyError
            If `Filename` is a valid h5 file but doesn't have a 'Data' field
        IOError
            If `Filename` doesn't exist
        """
        
        return Serializer.load_from_hdf(filename)["Data"]
    
    @staticmethod
    def check_if_file_exists(filename):
        """Ensure that a file exists
        
        Parameters
        ----------
        Filename : str
            Path to resource to check
        
        Raises
        ------
        Exception
            If file already exists
        """
        
        if os.path.exists(filename):
            raise IOError("File %s already exists!" % filename)
    

    @staticmethod
    def SaveData(*args, **kwargs):
        msg = ('This function name is depricated, use save_data '
               'instead. This alias will be removed in version 2.7')
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return Serializer.save_data(*args, **kwargs)
                
    @staticmethod
    def LoadData(*args, **kwargs):
        msg = ('This function name is depricated, use load_data '
               'instead. This alias will be removed in version 2.7')
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return Serializer.load_data(*args, **kwargs)
    
    def SaveToHDF(self, *args, **kwargs):
        msg = ('This function name is depricated, use save_to_hdf '
               'instead. This alias will be removed in version 2.7')
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self.save_to_hdf(*args, **kwargs)
    
    @staticmethod
    def LoadFromHDF(*args, **kwargs):
        msg = ('This function name is depricated, use load_from_hdf '
               'instead. This alias will be removed in version 2.7')
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return Serializer.load_from_hdf(*args, **kwargs)
     
    @staticmethod  
    def SaveEntryAsEArray(*args, **kwargs):
        msg = ('This function name is depricated, use save_e_array '
               'instead. This alias will be removed in version 2.7')
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return Serializer.save_e_array(*args, **kwargs)
    
    @staticmethod
    def SaveEntryAsCArray(*args, **kwargs):
        msg = ('This function name is depricated, use save_c_array '
               'instead. This alias will be removed in version 2.7')
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return Serializer.save_c_array(*args, **kwargs)
    
    @staticmethod
    def SaveCSRMatrix(*args, **kwargs):
        msg = ('This function name is depricated, use save_csr_matrix '
               'instead. This alias will be removed in version 2.7')
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return Serializer.save_csr_matrix(*args, **kwargs)
    
    @staticmethod
    def LoadCSRMatrix(*args, **kwargs):
        msg = ('This function name is depricated, use load_csr_matrix '
               'instead. This alias will be removed in version 2.7')
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return Serializer.load_csr_matrix(*args, **kwargs)
    
    @staticmethod
    def CheckIfFileExists(*args, **kwargs):
        msg = ('This function name is depricated, use check_if_file_exists '
               'instead. This alias will be removed in version 2.7')
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return Serializer.check_if_file_exists(*args, **kwargs)
    