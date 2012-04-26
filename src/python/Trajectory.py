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
from msmbuilder import Conformation
from msmbuilder.Serializer import Serializer
from msmbuilder import xtc
from msmbuilder import dcd


MAXINT16=32766
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

class Trajectory(Conformation.ConformationBaseClass):
    """This is the representation of a sequence of  conformations.

    Notes:
    Use classmethod LoadFromPDB to create an instance of this class from a PDB filename.
    The Trajectory is a dictionary-like object.
    The dictionary key 'XYZList' contains a numpy array of XYZ coordinates, stored such that
    X[i,j,k] gives Frame i, Atom j, Coordinate k.
    """
    def __init__(self,S):
        """Create a Trajectory from a single conformation.  Leave the XYZList key as an empty list for easy appending."""
        Conformation.ConformationBaseClass.__init__(self,S)
        self["XYZList"]=[]
        if "XYZList" in S: self["XYZList"]=S["XYZList"].copy()
    
    #
    # I don't think these are necessary anymore
    #
    
    #def RestrictAtomIndices(self,AtomIndices):
    #    Conformation.ConformationBaseClass.RestrictAtomIndices(self,AtomIndices)
    #    self["XYZList"]=self["XYZList"][:,AtomIndices]
    
    #def Stride(self, stride):
    #    self["XYZList"] = self["XYZList"][0:len(self['XYZList']):stride]
    #    return self
        
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
    
    def SaveToLHDF(self,Filename,Precision=default_precision):
        """Save a Trajectory instance to a Lossy HDF File.  First, remove the XYZList key because it should be written using the special CArray operation.  This file format is roughly equivalent to an XTC and should comparable file sizes but with better IO performance."""
        Serializer.CheckIfFileExists(Filename)
        key="XYZList"
        X=self.pop(key)
        Serializer.SaveToHDF(self,Filename)
        Rounded=_ConvertToLossyIntegers(X,Precision)
        self[key]=Rounded
        Serializer.SaveEntryAsEArray(self[key],key,Filename=Filename)
        self[key]=X
        
    def SaveToXTC(self,Filename,Precision=default_precision):
        """Take a Trajectory instance and dump the coordinates to XTC"""
        Serializer.CheckIfFileExists(Filename)
        XTCFile=xtc.XTCWriter(Filename)
        for i in range(len(self["XYZList"])):
            XTCFile.write(self["XYZList"][i],1,i,np.eye(3,3,dtype='float32'),Precision)
        
    def SaveToPDB(self,Filename):
        """Write a conformation as a PDB file."""
        for i in range(len(self["XYZList"])):
            PDB.WritePDBConformation(Filename,self["AtomID"], self["AtomNames"],self["ResidueNames"],self["ResidueID"],self["XYZList"][i],self["ChainID"])

    def SaveToXYZ(self,Filename):
        """Write a conformation as a XYZ file."""
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
        """Auto-detect format and save."""
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
        try:
            self["XYZList"]=self["XYZList"].tolist()
        except:
            pass
        C1=Conformation.Conformation.LoadFromPDB(Filename)
        Temp=C1["XYZ"]
        if len(Temp)!=len(self["XYZList"][0]):
            raise NameError("Tried to add wrong number of coordinates.")
        else:
            self["XYZList"].append(Temp)
    
    @classmethod
    def LoadFromPDB(cls,Filename):       
        """Create a Trajectory from a PDB Filename."""
        return(Trajectory(PDB.LoadPDB(Filename,AllFrames=True)))
    
    @classmethod
    def LoadFromXTC(cls,XTCFilenameList,PDBFilename=None,Conf=None,PreAllocate=True,JustInspect=False):
        """Create a Trajectory from a Filename."""
        if PDBFilename!=None:
            A=Trajectory.LoadFromPDB(PDBFilename)
        elif Conf!=None:
            A=Trajectory(Conf)
        else:
            raise Exception("ERROR: Need a conformation to construct a trajectory.")
        if not JustInspect:
            A["XYZList"]=[]
            for c in xtc.XTCReader(XTCFilenameList):
                A["XYZList"].append(np.array(c.coords).copy())
            A["XYZList"]=np.array(A["XYZList"])
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
    def LoadFromHDF(cls,Filename,JustInspect=False):
        """Load a conformation that was previously saved as HDF."""
        if not JustInspect:
            S=Serializer.LoadFromHDF(Filename)
            A=cls(S)
            return(A)
        else:
            F1=tables.File(Filename)
            Shape=F1.root.XYZList.shape
            F1.close()
            return(Shape)
    
    
    @classmethod
    def LoadFromLHDF(cls,Filename,JustInspect=False,Precision=default_precision):
        """Load a conformation that was previously saved as HDF."""
        if not JustInspect:
            S=Serializer.LoadFromHDF(Filename)
            A=cls(S)
            A["XYZList"]=_ConvertFromLossyIntegers(A["XYZList"],Precision)
            return(A)
        else:
            F1=tables.File(Filename)
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
    def ReadFrame(cls,TrajFilename,WhichFrame,Conf=None):
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
            raise IOError("Incorrect file type--cannot get conformation %s"%TrajFilename)
    
    
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
    def AppendFramesToFile(cls,filename,XYZList,precision=default_precision):
        """Append an array of XYZ data to an existing .h5 or .lh5 file.
        """

        extension = os.path.splitext(filename)[1]

        if extension in [".h5", ".lh5"]:
            File = tables.File(filename,"a")
        else:
            raise(Exception("File must be .h5 or .lh5"))

        if not File.root.XYZList.shape[1:] == XYZList.shape[1:]:
            raise(Exception("Error: data cannot be appended to trajectory due to incorrect shape."))
        
        if extension == ".h5":
            File.root.XYZList.append(XYZList)

        elif extension == ".lh5":
            z = _ConvertToLossyIntegers(XYZList,precision)
            File.root.XYZList.append(z)
        File.flush()
        File.close()
