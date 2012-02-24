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

""" This is a library of functions that calculate useful geometric properties of
conformations and trajectories. This includes contact maps, residue-residue
distances, etc. """

import numpy as np

from msmbuilder import Conformation
from msmbuilder import Trajectory

# General Functions
def ProcessXYZ(X1):
    """ A helper function allowing one to pass an argument as either a
    Trajectory or Conformation class. Not very elegant, but effective.

    Argument: X1  - is a Trajectory or Conformation object
    Returns:  xyz - an array of xyz coordinates [ [x,y,z], [], ... ]"""

    # Check if input is a Trajectory. If so, do nothing.
    if "XYZList" in X1.keys():
        xyz = X1["XYZList"]
    elif "XYZ" in X1.keys():
        xyz = np.array([ X1["XYZ"] ])
    else:
        raise LookupError
    return xyz

def L2_norm(xyz1, xyz2, axis=0):
    """Calculates the L2 distance between two array conformations using the  specified axis.
    
    Arguments:
    xyz1 - array of coordinates
    xyz2 - array of coordinates
    axis - which axis to compute distance between

    Returns:
    distance (float)
    """
    return np.power( np.sum( np.power( xyz1-xyz2, 2.0), axis=axis ), 0.5 )

def BetaResidueDist(X1, i, j):
    """Get the distance between residues i and j.  For residues with beta
    carbons, use the beta carbon distance.  Otherwise use alpha carbons.

    Arguments:
    X1  - is a Trajectory or Conformation object
    i,j - indices of the two residues (zero index)

    Returns:
    dist (float) - the residue-residue distance, in nm """
    
    xyz = ProcessXYZ(X1)
    r1 = X1["ResidueNames"][X1["IndexList"][i][0]]
    r2 = X1["ResidueNames"][X1["IndexList"][j][0]]
    a1 = "CB"
    a2 = "CB"
    if r1=="GLY": a1="CA"
    if r2=="GLY": a2="CA"
        
    i0 = np.where( (X1["AtomNames"] == a1) & (X1["ResidueID"] == i+1) )
    i1 = np.where( (X1["AtomNames"] == a2) & (X1["ResidueID"] == j+1) )
    assert len( i0[0] ) != 0
    assert len( i1[0] ) != 0
    
    dist = L2_norm( xyz[i0], xyz[i1] )
    return dist.flatten()

def AtomicContactMap(C1, atomInds, cutoff):
    """Get the contact map for the atoms listed in atomInds.
    From Gregory R. Bowman.

    Arguments:
    C1  - is a Conformation object
    atomInds (array)  - indices of atoms to use
    cutoff (float) - distance within which to consider atoms in contact (default msmbuilder units are Angstroms)

    Returns:
    map (array) - contact map with a 1 in position i,j if atomInds[i] and atomInds[j] are within the cutoff distance of each other."""

    xyz = ProcessXYZ(C1)[0]
    nAtoms = xyz.shape[0]

    # check that atom indices in acceptable range
    if atomInds.min()<0 or atomInds.max()>nAtoms:
        raise IndexError

    n = atomInds.shape[0]
    contactMap = np.ones((n,n))
    for i in xrange(n):
        for j in xrange(i+1,n):
            d = L2_norm(xyz[atomInds[i]], xyz[atomInds[j]])
            if d > cutoff:
                contactMap[i,j] = 0
                contactMap[j,i] = 0

    return contactMap

def GetRg(XYZ):
    """Return the radius of gyration of a frame.

    Inputs:
    XYZ: XYZ coordinates (Numpy array)

    Output: Rg
    """
    mu=XYZ.mean(0)
    XYZ2=XYZ-np.tile(mu,(len(XYZ),1))
    Rg=(XYZ2**2.).mean()**0.5
    return Rg
