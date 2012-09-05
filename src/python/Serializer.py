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

"""The Serializer is a base class for storing dictionary-like objects to disk as HDF5 files.

Notes:
The Conformation, Trajectory, and Project all inherit Serializer for HDF5 IO.
"""

import tables
import os
import scipy.sparse
import numpy as np
import warnings

try:
    Filter=tables.Filters(complevel=9,complib='blosc',shuffle=True)
except:
    warnings.warn("missing BLOSC, no compression will used.")
    Filter = tables.Filters()

class Serializer(dict):
    """
    A generic class for dumping dictionaries of data onto disk
    using the pytables HDF5 library.
    
    """
    
    def __init__(self,DictLikeObject=dict()):
        """Build Serializer from dictionary
        
        All Serializer subclass constructors take a dictionary-like object
        as input.  Subclasses will attempt to load (key,value) pairs by name,
        which will raise an exception if something is missing.
        
        Parameters
        ----------
        DictLikeObject : dict
            Data to populate the Serializer with
        
        """
        self.update(DictLikeObject)
        
    def SaveToHDF(self,Filename,loc="/", do_file_check=True):
        """Save the contents of the serializer to disk
        
        A generic function for saving ForceFields / Topologies / Conformations
        to H5 files.  Certain types of data cannot be stored as simple arrays,
        so these are the exceptions (if statements) in this function.
        
        Parameters
        ----------
        Filename : str
            Filename to save data to
        Loc : str, optional
            Resource root in HDF file
        do_file_check : bool, optional
            Check that the location is free before saving
        
        Raises
        ------
        Exception
            If `do_file_check` is True and something is already stored in `Filename`
            
        """
        
        # check h5 file doesn't already exist
        if do_file_check:
            self.CheckIfFileExists(Filename)
        
        F=tables.File(Filename,'a')
        
        for key,data in self.iteritems():
            #print(key,data)
            try:#This checks if the list is homogenous and can be stored in an array data type (ie a square tensor).  If not, we need VLArray
                TEMP=np.array(data)
                if TEMP.dtype==np.dtype("object"):
                    raise ValueError#This check is necessary for Numpy 1.6.0 and greater, which allow inhomogeneous lists to be converted to dtype arrays.                
            except ValueError:
                F.createVLArray(loc,key,tables.Int16Atom())
                for x in data:
                    F.getNode(loc,key).append(x)
                continue
            self.SaveEntryAsEArray(np.array(data),key,F0=F,loc=loc)
        F.flush()
        F.close()
    
    @classmethod
    def LoadFromHDF(cls,Filename,loc="/"):
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
        
        A=Serializer()
        F=tables.File(Filename,'r')
        for d in F.listNodes(loc):
            if type(d)==tables.VLArray:
                A.update([[d.name,d.read()]])
                continue
            if d[:].size==1:
                A[d.name]=d[:].item()
                continue
            if d.shape[0]==1:
                A[d.name]=np.array(d.read())
                continue
            A[d.name]=d[:]
        F.close()
        A['SerializerFilename'] = os.path.abspath(Filename)
        return(cls(A))
    
    @staticmethod
    def SaveEntryAsEArray(Data, Key, Filename=None, F0=None, loc="/"):
        """Save dict entry as compressed EArray
        
        E means extensible, which can be useful for extending trajectories.
        
        Parameters
        ----------
        Data : 
        Key : 
        Filename : 
        F0 : 
        loc : 
        
        """
        if F0==None and Filename==None:
            raise Exception("Must Specify either F or Filename")
        if F0==None:
            F=tables.File(Filename,'a')
        else:
            F=F0
        if np.rank(Data)==0:
            Data=np.array([Data])
        sh=np.array(np.shape(Data))
        sh[0]=0
        F.createEArray("/",Key,tables.Atom.from_dtype(Data.dtype),sh,filters=Filter)
        node=F.getNode("/",Key)
        node.append(Data)
        if F0==None:
            F.close()
    
    @staticmethod
    def SaveEntryAsCArray(Data, Key, Filename=None, F0=None, loc="/"):
        """Save this dictionary entry as a compressed CArray
        
        Parameters
        ----------
        Data : 
        Key : 
        Filename : 
        F0 : 
        loc : 
        
        Notes
        -----
        CArray tends to give about 20% better performance than EArray.
        Also, for VHP (576 atoms), Chunkshape[0]=8 seems to give perhaps another
        20%.  The total enhancement appears to be an total of 8800 conformations
        per second versus 6300 for EArray without chunkshape optimization.
        
        """
        if F0==None and Filename==None:
            raise Exception("Must Specify either F or Filename")
        if F0==None:
            F=tables.File(Filename,'a')
        else:
            F=F0
        if np.rank(Data)==0:
            Data=np.array([Data])
        F.createCArray("/",Key,tables.Atom.from_dtype(Data.dtype),np.shape(Data),filters=Filter)
        node=F.getNode("/",Key)
        node[:]=Data
        if F0==None:
            F.close()
    
    @staticmethod
    def SaveCSRMatrix(Filename, T):
        """Save a CSR sparse matrix to disk
        
        Parameters
        ----------
        Filename : str
            Location to save to
        T : csr_matrix
            matrix to save
        
        Raises
        ------
        TypeError
            If `T` is not a CSR sparse matrix
        Exception
            If a file exists at `Filename`
        """
        
        if not scipy.sparse.isspmatrix_csr(T):
            raise TypeError("A CSR sparse matrix is required for saving to .csr.")
        X=Serializer({"data":T.data,"indices":T.indices,"indptr":T.indptr,"Shape":T.shape})
        X.SaveToHDF(Filename)
        del X
    
    @staticmethod
    def LoadCSRMatrix(Filename):
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
        X=Serializer.LoadFromHDF(Filename)
        return scipy.sparse.csr_matrix((X["data"], X["indices"], X["indptr"]),
                                        shape=X["Shape"])
    
    @staticmethod
    def SaveData(Filename,Data):
        """Quickly dump an array to disk in h5 format
        
        Writes the array in the 'Data' field
        
        Parameters
        ----------
        Filename : str
            Path to resource to write to
        Data : ndarray
            numpy array to write to disk
        """
        
        X=Serializer({"Data":Data})
        X.SaveToHDF(Filename)
    
    @staticmethod
    def LoadData(Filename):
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
        Exception
            If `Filename` doesn't exist
        """
        
        D=Serializer.LoadFromHDF(Filename)
        return D["Data"]
    
    @staticmethod
    def CheckIfFileExists(Filename):
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
        
        if os.path.exists(Filename):
            raise Exception("Error: HDF5 File %s Already Exists!"%Filename)



