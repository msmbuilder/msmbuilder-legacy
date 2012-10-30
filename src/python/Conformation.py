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
from msmbuilder import pdb
from msmbuilder.utils import deprecated

class ConformationBaseClass(dict):
    """Base class for Trajectory and Conformation classes.  
    Not for separate use."""
    def __init__(self, dict_like=None):
        """Initialize object.  Optionally include data from a dictionary 
        object dict_like."""
        super(ConformationBaseClass, self).__init__(dict_like)
        
        keys_to_force_copy = ["ChainID", "AtomNames", "ResidueNames", "AtomID",
                              "ResidueID"]
        for key in keys_to_force_copy: # To avoid overwriting something
            self[key] = self[key].copy()
            
        self["ResidueNames"] = self["ResidueNames"].copy().astype("S4")
        self.update_index_list()

    def update_index_list(self):
        """Construct a list of which atoms belong to which residues.

        NOTE: these indices are NOT the same as the ResidueIDs--these indices 
            take the value 0, 1, ..., (n-1)
        where n is the number of residues.
        """
        self["IndexList"] = [ [] for i in range(self.num_residues) ]

        zero_index_residue_id = self.get_enumerated_residue_id()
        for i in range(self.num_atoms):
            self["IndexList"][zero_index_residue_id[i]].append(i)

    @property
    def num_atoms(self):
        """Return the number of atoms in this object."""
        return len(self["AtomNames"])

    @property
    def num_residues(self):
        """Return the number of residues in this object."""
        return len(np.unique(self["ResidueID"]))

    @deprecated()
    def get_number_of_atoms(self):
        """Return the number of atoms in this object."""
        return self.num_atoms

    @deprecated()
    def get_number_of_residues(self):
        """Return the number of residues in this object."""
        return self.num_residues

    def get_enumerated_atom_id(self):
        """Returns an array of consecutive integers that enumerate over all 
        atoms in the system.  STARTING WITH ZERO!"""
        return np.arange(len(self["AtomID"]))

    def get_enumerated_residue_id(self):
        """Returns an array of NONUNIQUE consecutive integers that enumerate 
        over all Residues in the system.  STARTING WITH ZERO!

        Note: This will return something like [0,0,0,1,1,1,2,2,2]--the first 3 
        atoms belong to residue 0, the next 3 belong to 1, etc.
        """
        unique_residue_ids = np.unique(self["ResidueID"])
        residue_id_dict = dict([ [x,i] for i,x in 
                                enumerate(unique_residue_ids)])
        
        residue_ids = np.zeros(len(self["ResidueID"]), 'int')
        for i in xrange(self.num_atoms):
            residue_ids[i] = residue_id_dict[self["ResidueID"][i]]
        return residue_ids

    def restrict_atom_indices(self, atom_indices):
        for key in ["AtomID", "ChainID", "ResidueID", "AtomNames",
                    "ResidueNames"]:
            self[key] = self[key][atom_indices]

        self.update_index_list()

class Conformation(ConformationBaseClass):
    """A single biomolecule conformation.  Use classmethod load_from_pdb to 
    create an instance of this class from a PDB filename"""    
    def __init__(self, dict_like):
        """Initializes object from a dictionary-like object (dict_like)."""
        ConformationBaseClass.__init__(self, dict_like)
        self["XYZ"] = dict_like["XYZ"].copy()

    def restrict_atom_indices(self, atom_indices):
        ConformationBaseClass.restrict_atom_indices(self, atom_indices)
        self["XYZ"] = self["XYZ"][atom_indices]
        
    def save_to_pdb(self, filename):
        """Write conformation as a PDB file."""
        pdb.write_pdb_conformation(filename, self["AtomID"], self["AtomNames"],
                                   self["ResidueNames"], self["ResidueID"],
                                   self["XYZ"], self["ChainID"])
        
    @classmethod
    def load_from_pdb(cls, filename):       
        """Create a conformation from a PDB File."""
        return cls(pdb.load_pdb(filename))

