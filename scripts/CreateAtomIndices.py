#!/usr/bin/env python
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

import sys
import numpy as np

from msmbuilder import Conformation
from msmbuilder import arglib
import logging
logger = logging.getLogger(__name__)

def run(PDBfn, atomtype):
  
  # dictionaries with residue types as keys and list of atoms to keep for given residue as entries
  toKeepDict = {
"ALA": ["N", "CA", "CB", "C", "O"],
"ACE": ["N", "CA", "CB", "C", "O"],
"CALA": ["N", "CA", "CB", "C", "O"],
"NALA": ["N", "CA", "CB", "C", "O"],
"ARG": ["N", "CA", "CB", "C", "O", "CG", "CD", "NE", "CZ"],
"CARG": ["N", "CA", "CB", "C", "O", "CG", "CD", "NE", "CZ"],
"NARG": ["N", "CA", "CB", "C", "O", "CG", "CD", "NE", "CZ"],
"ASN": ["N", "CA", "CB", "C", "O", "CG", "OD1", "ND2"],
"CASN": ["N", "CA", "CB", "C", "O", "CG", "OD1", "ND2"],
"NASN": ["N", "CA", "CB", "C", "O", "CG", "OD1", "ND2"],
"ASP": ["N", "CA", "CB", "C", "O", "CG"],
"CASP": ["N", "CA", "CB", "C", "O", "CG"],
"NASP": ["N", "CA", "CB", "C", "O", "CG"],
"CYS": ["N", "CA", "CB", "C", "O", "SG"],
"CYX": ["N", "CA", "CB", "C", "O", "SG"],
"CCYS": ["N", "CA", "CB", "C", "O", "SG"],
"NCYS": ["N", "CA", "CB", "C", "O", "SG"],
"GLU": ["N", "CA", "CB", "C", "O", "CG", "CD"],
"CGLU": ["N", "CA", "CB", "C", "O", "CG", "CD"],
"NGLU": ["N", "CA", "CB", "C", "O", "CG", "CD"],
"GLN": ["N", "CA", "CB", "C", "O", "CG", "CD", "OE1", "NE2"],
"CGLN": ["N", "CA", "CB", "C", "O", "CG", "CD", "OE1", "NE2"],
"NGLN": ["N", "CA", "CB", "C", "O", "CG", "CD", "OE1", "NE2"],
"GLY": ["N", "CA", "C", "O"],
"CGLY": ["N", "CA", "C", "O"],
"NGLY": ["N", "CA", "C", "O"],
"HSD": ["N", "CA", "CB", "C", "O", "CG", "ND1", "CE1", "NE2", "CD2"],
"HIS": ["N", "CA", "CB", "C", "O", "CG", "ND1", "CE1", "NE2", "CD2"],
"HID": ["N", "CA", "CB", "C", "O", "CG", "ND1", "CE1", "NE2", "CD2"],
"CHID": ["N", "CA", "CB", "C", "O", "CG", "ND1", "CE1", "NE2", "CD2"],
"NHID": ["N", "CA", "CB", "C", "O", "CG", "ND1", "CE1", "NE2", "CD2"],
"HIE": ["N", "CA", "CB", "C", "O", "CG", "ND1", "CE1", "NE2", "CD2"],
"CHIE": ["N", "CA", "CB", "C", "O", "CG", "ND1", "CE1", "NE2", "CD2"],
"NHIE": ["N", "CA", "CB", "C", "O", "CG", "ND1", "CE1", "NE2", "CD2"],
"HIP": ["N", "CA", "CB", "C", "O", "CG", "ND1", "CE1", "NE2", "CD2"],
"CHIP": ["N", "CA", "CB", "C", "O", "CG", "ND1", "CE1", "NE2", "CD2"],
"NHIP": ["N", "CA", "CB", "C", "O", "CG", "ND1", "CE1", "NE2", "CD2"],
"ILE": ["N", "CA", "CB", "C", "O", "CG1", "CG2", "CD"],
"CILE": ["N", "CA", "CB", "C", "O", "CG1", "CG2", "CD"],
"NILE": ["N", "CA", "CB", "C", "O", "CG1", "CG2", "CD"],
"LEU": ["N", "CA", "CB", "C", "O", "CG"],
"CLEU": ["N", "CA", "CB", "C", "O", "CG"],
"NLEU": ["N", "CA", "CB", "C", "O", "CG"],
"LYP": ["N", "CA", "CB", "C", "O", "CG", "CD", "CE", "NZ"],
"LYS": ["N", "CA", "CB", "C", "O", "CG", "CD", "CE", "NZ"],
"CLYP": ["N", "CA", "CB", "C", "O", "CG", "CD", "CE", "NZ"],
"NLYP": ["N", "CA", "CB", "C", "O", "CG", "CD", "CE", "NZ"],
"MET": ["N", "CA", "CB", "C", "O", "CG", "SD", "CE"],
"CMET": ["N", "CA", "CB", "C", "O", "CG", "SD", "CE"],
"NMET": ["N", "CA", "CB", "C", "O", "CG", "SD", "CE"],
"NME": ["N", "CA", "CB", "C", "O", "CG", "SD", "CE"],
"PHE": ["N", "CA", "CB", "C", "O", "CG", "CZ"],
"CPHE": ["N", "CA", "CB", "C", "O", "CG", "CZ"],
"NPHE": ["N", "CA", "CB", "C", "O", "CG", "CZ"],
"PRO": ["N", "CA", "CB", "C", "O", "CD", "CG"],
"CPRO": ["N", "CA", "CB", "C", "O", "CD", "CG"],
"NPRO": ["N", "CA", "CB", "C", "O", "CD", "CG"],
"SER": ["N", "CA", "CB", "C", "O", "OG"],
"CSER": ["N", "CA", "CB", "C", "O", "OG"],
"NSER": ["N", "CA", "CB", "C", "O", "OG"],
"THR": ["N", "CA", "CB", "C", "O", "CG2", "OG1"],
"CTHR": ["N", "CA", "CB", "C", "O", "CG2", "OG1"],
"NTHR": ["N", "CA", "CB", "C", "O", "CG2", "OG1"],
"TRP": ["N", "CA", "CB", "C", "O", "CG", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3", "CD2"],
"CTRP": ["N", "CA", "CB", "C", "O", "CG", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3", "CD2"],
"NTRP": ["N", "CA", "CB", "C", "O", "CG", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3", "CD2"],
"TYR": ["N", "CA", "CB", "C", "O", "CG", "CZ", "OH"],
"CTYR": ["N", "CA", "CB", "C", "O", "CG", "CZ", "OH"],
"NTYR": ["N", "CA", "CB", "C", "O", "CG", "CZ", "OH"],
"VAL": ["N", "CA", "CB", "C", "O"],
"CVAL": ["N", "CA", "CB", "C", "O"],
"NVAL": ["N", "CA", "CB", "C", "O"],
"NLE": ["N", "CA", "CB", "C", "O", "CG", "CD", "CE"],
"CNLE": ["N", "CA", "CB", "C", "O", "CG", "CD", "CE"],
"NNLE": ["N", "CA", "CB", "C", "O", "CG", "CD", "CE"],
"SOL": []
}

  if atomtype == 'heavy': pass

  elif atomtype == 'minimal':
    for key in toKeepDict.keys():
      toKeepDict[key] = ["N", "CA", "CB", "C", "O"]

  elif atomtype == 'alpha':
    for key in toKeepDict.keys():
      toKeepDict[key] = ["CA"]
  elif atomtype == "all":
    pass
  else:
      logger.error("Cannot understand atom type: %s", atomtype)
      sys.exit(1)

  C1 = Conformation.load_from_pdb(PDBfn)

  if atomtype != "all":
    IndicesToKeep = GrabSpecificAtoms(C1,toKeepDict)
  else:
    IndicesToKeep = np.arange(C1.GetNumberOfAtoms())
  
  return IndicesToKeep

def GrabSpecificAtoms(C1,toKeepDict):
  IndicesToKeep=[]
  for k,CurrentIndices in enumerate(C1["IndexList"]):
      Residue = C1["ResidueNames"][CurrentIndices[0]]
      DesiredAtoms = toKeepDict[Residue]
      IndicesRelativeToCurrentResidue = np.where(np.in1d(C1["AtomNames"][CurrentIndices],DesiredAtoms)==True)[0]
      IndicesToKeep.extend(np.array(CurrentIndices)[IndicesRelativeToCurrentResidue])
  IndicesToKeep = np.array(IndicesToKeep,'int')
  return(IndicesToKeep)
  
if __name__ == "__main__":
  parser = arglib.ArgumentParser(description="Creates an atom indices file from a PDB.")
  parser.add_argument('pdb')
  parser.add_argument('output', default='AtomIndices.dat')
  parser.add_argument('atom_type', help='''Atoms to include in index file.
    One of four options: (1) minimal (CA, CB, C, N, O, recommended), (2) heavy,
    (3) alpha (carbons), or (4) all.  Use "all" in cases where protein
    nomenclature may be inapproprate, although you may want to define your own
    indices in such situations.''', choices=['minimal', 'heavy', 'alpha', 'all'],
    default='minimal')
  args = parser.parse_args()
  arglib.die_if_path_exists(args.output)

  indices = run(args.pdb, args.atom_type)
  
  np.savetxt(args.output, indices, '%d')
  logger.info('Saved output to %s', args.output)
