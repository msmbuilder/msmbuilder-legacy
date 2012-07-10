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

"""Contains classes for dealing with conformations.
"""
import numpy as np
from msmbuilder import PDB, Serializer

class ConformationBaseClass(Serializer):
    """Base class for Trajectory and Conformation classes.  Not for separate use."""
    def __init__(self,DictLike=None):
        """Initialize object.  Optionally include data from a dictionary object DictLike."""
        Serializer.__init__(self,DictLike)
        KeysToForceCopy=["ChainID","AtomNames","ResidueNames","AtomID","ResidueID"]
        for key in KeysToForceCopy:#Force copy to avoid owning same numpy memory.
            self[key]=self[key].copy()
            
        self["ResidueNames"]=self["ResidueNames"].copy().astype("S4")
        self.UpdateIndexList()

    def UpdateIndexList(self):
        """Construct a list of which atoms belong to which residues.

        NOTE: these indices are NOT the same as the ResidueIDs--these indices take the value 0, 1, ..., (n-1)
        where n is the number of residues.
        """
        self["IndexList"]=[[] for i in range(self.GetNumberOfResidues())]

        ZeroIndexResidueID=self.GetEnumeratedResidueID()
        for i in range(self.GetNumberOfAtoms()):
            self["IndexList"][ZeroIndexResidueID[i]].append(i)

    def GetNumberOfAtoms(self):
        """Return the number of atoms in this object."""
        return len(self["AtomNames"])
    
    def GetNumberOfResidues(self):
        """Return the number of residues in this object."""
        return len(np.unique(self["ResidueID"]))

    def GetEnumeratedAtomID(self):
        """Returns an array of consecutive integers that enumerate over all atoms in the system.  STARTING WITH ZERO!"""
        return np.arange(len(self["AtomID"]))

    def GetEnumeratedResidueID(self):
        """Returns an array of NONUNIQUE consecutive integers that enumerate over all Residues in the system.  STARTING WITH ZERO!

        Note: This will return something like [0,0,0,1,1,1,2,2,2]--the first 3 atoms belong to residue 0, the next 3 belong to 1, etc.
        """
        UniquePDBID=np.unique(self["ResidueID"])
        D=dict([[x,i] for i,x in enumerate(UniquePDBID)])
        
        X=np.zeros(len(self["ResidueID"]),'int')
        for i in xrange(self.GetNumberOfAtoms()):
            X[i]=D[self["ResidueID"][i]]
        return X

    def RestrictAtomIndices(self,AtomIndices):
        for key in ["AtomID","ChainID","ResidueID","AtomNames","ResidueNames"]:
            self[key]=self[key][AtomIndices]

        self.UpdateIndexList()

class Conformation(ConformationBaseClass):
    """A single biomolecule conformation.  Use classmethod LoadFromPDB to create an instance of this class from a PDB filename"""    
    def __init__(self,S):
        """Initializes object from a dictionary-like object S."""
        ConformationBaseClass.__init__(self,S)
        self["XYZ"]=S["XYZ"].copy()

    def RestrictAtomIndices(self,AtomIndices):
        ConformationBaseClass.RestrictAtomIndices(self,AtomIndices)
        self["XYZ"]=self["XYZ"][AtomIndices]
        
    def SaveToPDB(self,Filename):
        """Write conformation as a PDB file."""
        PDB.WritePDBConformation(Filename,self["AtomID"], self["AtomNames"],self["ResidueNames"],self["ResidueID"],self["XYZ"],self["ChainID"])
        
    @classmethod
    def LoadFromPDB(cls,Filename):       
        """Create a conformation from a PDB File."""
        return(cls(PDB.LoadPDB(Filename)))

