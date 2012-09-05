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

from msmbuilder import PDB
from msmbuilder import Serializer
from msmbuilder import ConformationBaseClass, Conformation
from msmbuilder import xtc
from msmbuilder import dcd
import warnings

MAXINT16=np.iinfo(np.int16).max
MAXINT32=np.iinfo(np.int32).max
default_precision = 1000

def _ConvertToLossyIntegers(X,Precision):
    """Implementation of the lossy compression used in Gromacs XTC using the pytables library.  Convert 32 bit floats into 16 bit integers.  These conversion functions have been optimized for memory use.  Further memory reduction would require an in-place astype() operation, which one could create using ctypes."""
    if np.max(X)*float(Precision)< MAXINT16 and np.min(X)*float(Precision) > -MAXINT16:
        X*=float(Precision)
        Rounded=X.astype("int16")
        X/=float(Precision)
    else:
        X*=float(Precision)
        Rounded=X.astype("int32")
        X/=float(Precision)
        print("Data range too large for int16: try removing center of mass motion, check for 'blowing up, or just use .h5 or .xtc format.'")
    return(Rounded)

def _ConvertFromLossyIntegers(X,Precision):
    """Implementation of the lossy compression used in Gromacs XTC using the pytables library.  Convert 16 bit integers into 32 bit floats."""
    X2=X.astype("float32")
    X2/=float(Precision)
    return(X2)

class Trajectory(ConformationBaseClass):
    """This is the representation of a sequence of  conformations.

    Notes:
    Use classmethod LoadFromPDB to create an instance of this class from a PDB filename.
    The Trajectory is a dictionary-like object.
    The dictionary key 'XYZList' contains a numpy array of XYZ coordinates, stored such that
    X[i,j,k] gives Frame i, Atom j, Coordinate k.
    """
    def __init__(self,S):
        """Create a Trajectory from a single conformation.
        
        Leaves the XYZList key as an empty list for easy appending.
        
        Parameters
        ----------
        S : Conformation
            The single conformation to build a trajectory from
        
        Notes
        -----
        You probably want to use something else to load trajectories
        
        See Also
        --------
        LoadFromPDB
        LoadFromHDF
        LoadFromLHDF
        """
        
        ConformationBaseClass.__init__(self,S)
        self["XYZList"]=[]
        if "XYZList" in S: self["XYZList"]=S["XYZList"].copy()
    
        
    def subsample(self,stride):
        """Keep only the frames at some interval
        
        Shorten trajectory by taking every `stride`th frame. Note that this is
        an inplace operation!
        
        Parameters
        ----------
        stride : int
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
        if isinstance(key, int) or isinstance(key, slice) or isinstance(key,np.ndarray):
            if isinstance(key, int):
                key = [key]
            newtraj = copy.copy(self)
            newtraj['XYZList'] = self['XYZList'][key]
            return newtraj
        return super(Trajectory, self).__getitem__(key)
    
    def __len__(self):
        return len(self['XYZList'])

    def __add__(self,other):
        # Check type of other
        if not isinstance(other,Trajectory):
            raise TypeError('You can only add two Trajectory instances')
        Sum = copy.deepcopy(self)
        # Simply copy the XYZList in here if the Trajectory is Empty.
        if 'XYZList' not in self and 'XYZList' not in other:
            pass
        elif 'XYZList' not in self:
            Sum['XYZList'] = copy.deepcopy(other['XYZList'])
        elif 'XYZList' not in other:
            Sum['XYZList'] = copy.deepcopy(self['XYZList'])
        else:
            if not self['XYZList'].shape[1] == other['XYZList'].shape[1]:
                raise TypeError('The two trajectories don\'t have the same number of atoms')
            Sum['XYZList'] = np.vstack((self['XYZList'],other['XYZList']))
        return Sum
 
    def __iadd__(self,other):
        # Check type of other
        if not isinstance(other,Trajectory):
            raise TypeError('You can only add two Trajectory instances')
        # Simply copy the XYZList in here if the Trajectory is Empty.
        if 'XYZList' not in self:
            self['XYZList'] = copy.deepcopy(other['XYZList'])
        else:
            # Check number of atoms.
            if not self['XYZList'].shape[1] == other['XYZList'].shape[1]:
                raise TypeError('The two trajectories don\'t have the same number of atoms')
            self['XYZList'] = np.vstack((self['XYZList'],other['XYZList']))
        return self
 
    def SaveToLHDF(self,Filename,Precision=default_precision):
        """Save a Trajectory instance to a Lossy HDF File.
        
        First, remove the XYZList key because it should be written using the
        special CArray operation.  This file format is roughly equivalent to
        an XTC and should comparable file sizes but with better IO performance.
        
        Parameters
        ----------
        Filename : str
            location to save to
        Precision : float, optional
            I'm not really sure what this does (RTM 6/27).
        """
        Serializer.CheckIfFileExists(Filename)
        key="XYZList"
        X=self.pop(key)
        Serializer.SaveToHDF(self,Filename)
        Rounded=_ConvertToLossyIntegers(X,Precision)
        self[key]=Rounded
        Serializer.SaveEntryAsEArray(self[key],key,Filename=Filename)
        self[key]=X
        
    def SaveToXTC(self,Filename,Precision=default_precision):
        """Dump the coordinates to XTC
        
        Parameters
        ----------
        Filename : str
            location to save to
        Precision : float, optional
            I'm not really sure what this does (RTM 6/27).
        """
 
        Serializer.CheckIfFileExists(Filename)
        XTCFile=xtc.XTCWriter(Filename)
        for i in range(len(self["XYZList"])):
            XTCFile.write(self["XYZList"][i],1,i,np.eye(3,3,dtype='float32'),Precision)
        
    def SaveToPDB(self,Filename):
        """Dump the coordinates to PDB
        
        Parameters
        ----------
        Filename : str
            location to save to
            
        Notes
        -----
        Don't use this for a very big trajectory. PDB is plaintext and takes a lot
        of memory
        """
        
        for i in range(len(self["XYZList"])):
            PDB.WritePDBConformation(Filename,self["AtomID"], self["AtomNames"],self["ResidueNames"],self["ResidueID"],self["XYZList"][i],self["ChainID"])

    def SaveToXYZ(self,Filename):
        """Dump the coordinates to XYZ format
        
        Parameters
        ----------
        Filename : str
            location to save to
            
        Notes
        -----
        TODO: What exactly is the XYZ format? (RTM 6/27)
        """
        
        Answer = []
        Title = 'From MSMBuilder SaveToXYZ funktion'
        for i in range(len(self["XYZList"])):
            xyz = self["XYZList"][i]
            na = xyz.shape[0]
            Answer.append('%i' % na)
            Answer.append(Title)
            for j in range(xyz.shape[0]):
                Answer.append("%-5s% 12.6f% 12.6f% 12.6f" % (self["AtomNames"][j],xyz[j,0],xyz[j,1],xyz[j,2]))
        for i in range(len(Answer)):
            Answer[i] += "\n"
        with open(Filename,'w') as f: f.writelines(Answer)

    def Save(self,Filename,Precision=default_precision):
        """Dump the coordinates to disk in format auto-detected by filename
        
        Parameters
        ----------
        Filename : str
            location to save to
            
        Notes
        -----
        Formats supported are h5, xtc, pdb, lh5 and xyz
        """
        
        extension = os.path.splitext(Filename)[1]
        
        if extension == '.h5':
            self.SaveToHDF(Filename)
        elif extension == '.xtc':
            self.SaveToXTC(Filename)
        elif extension == '.pdb':
            self.SaveToPDB(Filename)
        elif extension == '.lh5':
            self.SaveToLHDF(Filename,Precision=Precision)
        elif extension == '.xyz':
            self.SaveToXYZ(Filename)
        else:
            raise IOError("File: %s. I don't understand the extension '%s'" % (Filename, extension))
                    
    def AppendPDB(self,Filename):
        """Add on to a pdb file
        
        Parameters
        ----------
        Filename : str
            location to save to
        
        """
        try:
            self["XYZList"]=self["XYZList"].tolist()
        except:
            pass
        C1=Conformation.LoadFromPDB(Filename)
        Temp=C1["XYZ"]
        if len(Temp)!=len(self["XYZList"][0]):
            raise NameError("Tried to add wrong number of coordinates.")
        else:
            self["XYZList"].append(Temp)

    
    @classmethod
    def LoadFromPDB(cls,Filename):       
        """Create a Trajectory from a PDB Filename
        
        Parameters
        ----------
        Filename : str
            location to load from
        """
        return(Trajectory(PDB.LoadPDB(Filename,AllFrames=True)))

    
    @classmethod
    def LoadFromXTC(cls, XTCFilenameList, PDBFilename=None, Conf=None, PreAllocate=True,
                    JustInspect=False, discard_overlapping_frames=False):
        """Create a Trajectory from a collection of XTC files

        Parameters
        ----------
        XTCFilenameList : list
            list of files to load from
        PDBFilename : str, optional
            XTC format doesnt have the connectivity information, which needs to be
            supplied. You can either supply it by giving a path to the PDB file (here)
            or by suppling a Conf or Traj object containing the right connectivity
            (next arg)
        Conf : Conformation, optional
            A conformation (actually passing another trajectory will work) that has
            the right atom labeling
        PreAlloc : bool, optional
            This doesnt do anything
        JustInspect : bool, optional
            Dont actually load, just return dimensions
        discard_overallping_frames : bool, optional
            Check for redundant frames and discard them. (RTM 6/27 should this be default True?)
            
        Returns
        -------
        Trajectory : Trajectory
            Trajectory loaded from disk. OR, if you supplied `just_inspect`=True,
            then just the shape
        """        
        if PDBFilename!=None:
            A=Trajectory.LoadFromPDB(PDBFilename)
        elif Conf!=None:
            A=Trajectory(Conf)
        else:
            raise Exception("ERROR: Need a conformation to construct a trajectory.")
        
        if not JustInspect:
            
            A["XYZList"] = []
            num_redundant = 0
            
            for i,c in enumerate( xtc.XTCReader(XTCFilenameList) ):

                # check to see if we have redundant frames as we load them up
                if discard_overlapping_frames:
                    if i > 0:
                        if np.sum( np.abs( c.coords - A["XYZList"][-1] ) ) < 10.**-8:
                            num_redundant += 1
                    A["XYZList"].append( np.array(c.coords).copy() )
                        
                else:
                    A["XYZList"].append( np.array(c.coords).copy() )
                    
            A["XYZList"]=np.array( A["XYZList"] )
            if num_redundant != 0:
                print "Found and discarded %d redunant snapshots in loaded traj" % num_redundant

        # in inspection mode
        else:
            i=0
            for c in xtc.XTCReader(XTCFilenameList):
                if i==0:
                    ConfShape=np.shape(c.coords)
                i=i+1
            Shape=np.array((i,ConfShape[0],ConfShape[1]))
            return(Shape)

        return(A)

    
    @classmethod
    def LoadFromDCD(cls,FilenameList,PDBFilename=None,Conf=None,PreAllocate=True,JustInspect=False):       
        """Create a Trajectory from a Filename."""
        
        if PDBFilename!=None:
            A=Trajectory.LoadFromPDB(PDBFilename)
        elif Conf!=None:
            A=Trajectory(Conf)
        else:
            raise Exception("ERROR: Need a conformation to construct a trajectory.")
        
        if not JustInspect:
            A["XYZList"]=[]
            for c in dcd.DCDReader(FilenameList):
                A["XYZList"].append(c.copy())
            A["XYZList"]=np.array(A["XYZList"])
        else: #This is wasteful to read everything in just to get the length
            XYZ=[]
            for c in dcd.DCDReader(FilenameList):
                XYZ.append(c.copy())
            XYZ=np.array(XYZ)
            return(XYZ.shape)

        return(A)
    
    @classmethod
    def LoadFromTRR(cls,TRRFilenameList,PDBFilename=None,Conf=None,PreAllocate=True,JustInspect=False):       
        """Create a Trajectory with title Title from a Filename."""
        if PDBFilename!=None:
            A=Trajectory.LoadFromPDB(PDBFilename)
        elif Conf!=None:
            A=Trajectory(Conf)
        else:
            raise Exception("ERROR: Need a conformation to construct a trajectory.")
        if not JustInspect:
            A["XYZList"]=[]
            A["Velocities"]=[]
            A["Forces"]=[]
            for c in xtc.TRRReader(TRRFilenameList):
                A["XYZList"].append(np.array(c.coords).copy())
                A["Velocities"].append(np.array(c.velocities).copy())
                A["Forces"].append(np.array(c.forces).copy())
            A["XYZList"]=np.array(A["XYZList"])
            A["Velocities"]=np.array(A["Velocities"])
            A["Forces"]=np.array(A["Forces"])
        else:
            i=0
            for c in xtc.TRRReader(TRRFilenameList):
                if i==0:
                    ConfShape=np.shape(c.coords)
                i=i+1
            Shape=np.array((i,ConfShape[0],ConfShape[1]))
            return(Shape)            
        return(A)
    

    @classmethod
    def LoadFromPDBList(cls,Filenames):       
        """Create a Trajectory with title Title from a Filename."""
        A=Trajectory.LoadFromPDB(Filenames[0])
        for f in Filenames[1:]:
            A.AppendPDB(f)
        A["XYZList"]=np.array(A["XYZList"])
        return(A)
    
    @classmethod
    def EnumChunksFromHDF(cls,TrajFilename,Stride=None,AtomIndices=None,ChunkSize=100000):

        """Variation of the generic function to load HDF files to dict-like object
        which enumerates chunks of the XYZList. This only really makes sense for 
        trajectory objects.
        ----------
        Filename : str
            Path to resource on disk to load from
        loc : str, option
            Resource root in HDF file
        
        """
        RestrictAtoms = False
        SubsampleXYZList = False
        if AtomIndices != None:
            RestrictAtoms = True
        if Stride != None:
            while ChunkSize % Stride != 0: # Need to do this in order to make sure we stride correctly.
                                            # since we read in chunks, and then we need the strides
                                            # to line up
                ChunkSize -= 1

        A=Serializer()
        F=tables.File(TrajFilename,'r')
        # load all the data other than XYZList

        if RestrictAtoms:
            A['AtomID'] = np.array( F.root.AtomID[AtomIndices], dtype=np.int32 )
            A['AtomNames'] = np.array( F.root.AtomNames[AtomIndices] )
            A['ChainID'] = np.array( F.root.ChainID[AtomIndices])
            A['ResidueID'] = np.array( F.root.ResidueID[AtomIndices], dtype=np.int32 )
            A['ResidueNames'] = np.array( F.root.ResidueNames[AtomIndices] )

            # IndexList is a VLArray, so we need to read the whole list with node.read() (same as node[:]) and then loop through each
                # row (residue) and remove the atom indices that are not wanted
            A['IndexList'] = [ [ i for i in row if (i in AtomIndices) ] for row in F.root.IndexList[:] ]
            
        else:
            A['AtomID'] = np.array( F.root.AtomID[:], dtype=np.int32 )
            A['AtomNames'] = np.array( F.root.AtomNames[:] )
            A['ChainID'] = np.array( F.root.ChainID[:])
            A['ResidueID'] = np.array( F.root.ResidueID[:], dtype=np.int32 )
            A['ResidueNames'] = np.array( F.root.ResidueNames[:] )
            
            A['IndexList'] = F.root.IndexList[:]

        A['SerializerFilename'] = os.path.abspath(TrajFilename)

        # Loaded everything except XYZList

        Shape = F.root.XYZList.shape
        begin_range_list = np.arange(0,Shape[0],ChunkSize) 
        end_range_list = np.concatenate( (begin_range_list[1:], [Shape[0]]) )

        for r0,r1 in zip( begin_range_list, end_range_list ):

            if RestrictAtoms:
                A['XYZList'] = np.array( F.root.XYZList[ r0 : r1 : Stride, AtomIndices ] )
            else:
                A['XYZList'] = np.array( F.root.XYZList[ r0 : r1 : Stride ] )

            yield cls(A)

        F.close()

        return

    @classmethod
    def EnumChunksFromLHDF(cls, TrajFilename, Precision=default_precision, Stride=None, AtomIndices=None, ChunkSize=100000):
        
        for A in cls.EnumChunksFromHDF( TrajFilename, Stride, AtomIndices, ChunkSize ):
            A['XYZList'] = _ConvertFromLossyIntegers( A['XYZList'], Precision )
            yield A

        return

    @classmethod
    def LoadFromHDF(cls, TrajFilename, JustInspect=False, Stride=None, AtomIndices=None ):
        
        if not JustInspect:
            A = list( cls.EnumChunksFromHDF( TrajFilename, Stride=Stride, AtomIndices=AtomIndices, ChunkSize=MAXINT32 ) )[0]
            return A

        else:
            F1=tables.File(TrajFilename)
            Shape=F1.root.XYZList.shape
            F1.close()
            return(Shape)

    @classmethod        
    def LoadFromLHDF(cls, TrajFilename, JustInspect=False, Precision=default_precision, Stride=None, AtomIndices=None ):

        if not JustInspect:
            A = cls.LoadFromHDF( TrajFilename, Stride=Stride, AtomIndices=AtomIndices )
            A['XYZList'] = _ConvertFromLossyIntegers( A['XYZList'], Precision )
            return A

        else:
            F1=tables.File(TrajFilename)
            Shape=F1.root.XYZList.shape
            F1.close()
            return(Shape)

    @classmethod
    def ReadXTCFrame(cls,TrajFilename,WhichFrame):
        """Read a single frame from XTC trajectory file without loading file into memory."""
        i=0
        for c in xtc.XTCReader(TrajFilename):
            if i == WhichFrame:
                return(np.array(c.coords))
            i = i+1
        raise Exception("Frame %d not found in file %s; last frame found was %d"%(WhichFrame,cls.TrajFilename,i))
    
    @classmethod
    def ReadDCDFrame(cls, TrajFilename, WhichFrame):
        """Read a single frame from DCD trajectory without loading file into memory."""
        reader = dcd.DCDReader(TrajFilename, firstframe=WhichFrame, lastframe=WhichFrame)
        xyz = None
        for c in reader:
            xyz = c.copy()
        if xyz == None:
            raise Exception("Frame %s not found in file %s." % (WhichFrame, TrajFilename))

        return xyz
        
    
    @classmethod
    def ReadHDF5Frame(cls,TrajFilename,WhichFrame):
        """Read a single frame from HDF5 trajectory file without loading file into memory."""
        F1=tables.File(TrajFilename)
        XYZ=F1.root.XYZList[WhichFrame]
        F1.close()
        return(XYZ)
    
    
    @classmethod
    def ReadLHDF5Frame(cls,TrajFilename,WhichFrame,Precision=default_precision):
        """Read a single frame from Lossy LHDF5 trajectory file without loading file into memory."""
        F1=tables.File(TrajFilename)
        XYZ=F1.root.XYZList[WhichFrame]
        F1.close()
        XYZ=_ConvertFromLossyIntegers(XYZ,Precision)
        return(XYZ)
    
    
    @classmethod
    def ReadFrame(cls, TrajFilename, WhichFrame, Conf=None):
        extension = os.path.splitext(TrajFilename)[1]
    
        if extension == '.xtc':
            return(Trajectory.ReadXTCFrame(TrajFilename, WhichFrame))
        elif extension == '.h5':
            return(Trajectory.ReadHDF5Frame(TrajFilename, WhichFrame))
        elif extension == '.lh5':
            return(Trajectory.ReadLHDF5Frame(TrajFilename, WhichFrame))
        elif extension == '.dcd':
            return(Trajectory.ReadDCDFrame(TrajFilename, WhichFrame))
        else:
            raise IOError("Incorrect file type--cannot get conformation %s" % TrajFilename)
    
    
    @classmethod
    def LoadTrajectoryFile(cls,Filename,JustInspect=False,Conf=None):
        """Loads a trajectory into memory, automatically deciding which methods to call based on filetype.  For XTC files, this method uses a pre-registered Conformation filename as a pdb."""
        extension = os.path.splitext(Filename)[1]
        
        if extension == '.h5':
            return Trajectory.LoadFromHDF(Filename,JustInspect=JustInspect)
            
        elif extension == '.xtc':
            if Conf==None:
                raise Exception("Need to register a Conformation to use XTC Reader.")
            return Trajectory.LoadFromXTC(Filename,Conf=Conf,JustInspect=JustInspect)
            
        elif extension == '.dcd':
            if Conf==None:
                raise Exception("Need to register a Conformation to use DCD Reader.")
            return Trajectory.LoadFromDCD(Filename,Conf=Conf,JustInspect=JustInspect)
            
        elif extension == '.lh5':
            return Trajectory.LoadFromLHDF(Filename,JustInspect=JustInspect)
            
        elif extension == '.pdb':
            return Trajectory.LoadFromPDB(Filename)
            
        else:
            raise IOError("File: %s. I don't understand the extension '%s'" % (Filename, extension))
    
    
    @classmethod
    def AppendFramesToFile(cls, filename, XYZList, precision=default_precision,
                           discard_overlapping_frames=False):
        """Append an array of XYZ data to an existing .h5 or .lh5 file.
        """

        extension = os.path.splitext(filename)[1]

        if extension in [".h5", ".lh5"]:
            File = tables.File(filename,"a")
        else:
            raise Exception("File must be .h5 or .lh5") 

        if not File.root.XYZList.shape[1:] == XYZList.shape[1:]:
            raise Exception("Error: data cannot be appended to trajectory due to incorrect shape.")

        
        if extension == ".h5":
            z = XYZList
        elif extension == ".lh5":
            z = _ConvertToLossyIntegers(XYZList,precision)

        # right now this only checks for the last written frame
        if discard_overlapping_frames:
            while (File.root.XYZList[-1,:,:] == z[0,:,:]).all():
                z = z[1:,:,:]

        File.root.XYZList.append(z)
        
        File.flush()
        File.close()


    @classmethod
    def _reduce_redundant_snapshots(cls, trajectory):
        """ Takes a trajectory object, and removes data from the 'XYZList' entry
            that are duplicated snapshots. Does this by checking if two contiguous
            snapshots are binary equivalent. """

        not_done = True
        i = 0 # index for the snapshot we're working on
        n = 0 # counts the number of corrections
        
        while not_done:

            # check to see if we are done
            if i == trajectory["XYZList"].shape[0]-1:
                break

            if (trajectory["XYZList"][i,:,:] == trajectory["XYZList"][i+1,:,:]).all():
                trajectory = trajectory[:i] + trajectory[i+1:]
                n += 1
            else:
                i += 1

        if n != 0:
            print "Warning: found and eliminated %d redundant snapshots in trajectory" % n
            
        return trajectory
