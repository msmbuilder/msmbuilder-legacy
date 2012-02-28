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

"""Tools for calculating RMSD distances.

Notes:
Please cite Haque, Beauchamp, Pande 2011 when using this RMSD tool.
"""

import numpy as np
from msmbuilder import rmsdcalc

def CheckCentered(XYZAtomMajor,Epsilon=1E-5):
    """Raise an exception if XYZAtomMajor has nonnzero center of mass(CM)."""
    XYZ=XYZAtomMajor.transpose(0,2,1)
    x=np.array([max(abs(XYZ[i].mean(0))) for i in xrange(len(XYZ))]).max()
    if x>Epsilon:
        raise Exception("The coordinate data does not appear to have been centered correctly.")

def centerConformations(XYZList):
    """Remove the center of mass from conformations.  Inplace to minimize mem. use."""
    for ci in xrange(XYZList.shape[0]):
        X=XYZList[ci].astype('float64')#To improve the accuracy of RMSD, it can help to do certain calculations in double precision.  This is _not_ one of those operations IMHO.
        X-=X.mean(0)
        XYZList[ci]=X.astype('float32')
    return

def calcGvalue(XYZ):
    """Calculate the sum of squares of the key matrix G.  A necessary component of Theobold RMSD algorithm."""
    conf=XYZ.astype('float64')#Doing this operation in double significantly improves numerical precision of RMSD
    G = 0
    G += np.dot(conf[:,0],conf[:,0])
    G += np.dot(conf[:,1],conf[:,1])
    G += np.dot(conf[:,2],conf[:,2])
    return G

class TheoData:
    """Stores temporary data required during Theobald RMSD calculation.

    Notes:
    Storing temporary data allows us to avoid re-calculating the G-Values repeatedly.
    Also avoids re-centering the coordinates.
    """
    
    def __init__(self,XYZData):
        """Create a container for intermediate values during RMSD Calculation.

        Notes:
        1.  We remove center of mass.
        2.  We pre-calculate matrix magnitudes (ConfG)
        """
        
        NumConfs=len(XYZData)
        NumAtoms=XYZData.shape[1]

        centerConformations(XYZData)

        NumAtomsWithPadding=4+NumAtoms-NumAtoms%4
    
        # Load data and generators into aligned arrays
        XYZData2 = np.zeros((NumConfs, 3, NumAtomsWithPadding), dtype=np.float32)
        for i in range(NumConfs):
            XYZData2[i,0:3,0:NumAtoms] = XYZData[i].transpose()

        #Precalculate matrix magnitudes
        ConfG = np.zeros((NumConfs,),dtype=np.float32)
        for i in xrange(NumConfs):
            ConfG[i] = calcGvalue(XYZData[i,:,:])

        self.XYZData=XYZData2
        self.G=ConfG
        self.NumAtoms=NumAtoms
        self.NumAtomsWithPadding=NumAtomsWithPadding
        self.CheckCentered()

    def CheckCentered(self):
        """Throw error if data not centered."""
        CheckCentered(self.GetData())
    
    def GetData(self):
        """Returns the XYZ coordinate data stored."""
        return(self.XYZData)

    def GetG(self):
        """Return the matrix magnitudes stored."""
        return(self.G)

    def SetData(self,XYZData,G):
        """Modify the data in self.

        Notes:
        For performance, this is done WITHOUT error checking.
        Only swap in data that is compatible (in shape) with previous data.
        """
        self.XYZData=XYZData
        self.G=G
        

class RMSDMetric:
    """Fast OpenMP Implementation of Theobald RMSD.

    Notes:
    """
    
    def PrepareData(self,XYZList):
        """Returns an object containing pre-processed data ready for RMSD calculation."""
        return TheoData(XYZList)

    def GetFastMultiDistance(self,Theo1,Theo2,Ind):
        """Calculate a vector of RMSDs between Theo1[Ind] and Theo2.

        Inputs:
        Theo1 -- A TheoData object.
        Theo2 -- A TheoData object.
        Ind -- The frame (of Theo1) to use.

        Notes:
        """
        return rmsdcalc.getMultipleRMSDs_aligned_T_g(
            Theo1.NumAtoms,
            Theo1.NumAtomsWithPadding,
            Theo1.NumAtomsWithPadding,
            Theo2.XYZData,
            Theo1.XYZData[Ind],
            Theo2.G,
            Theo1.G[Ind])
    
    def GetDistance(self,XYZ1,XYZ2):
        """Calculate the rmsd between frames XYZ1 and XYZ2.

        Notes:
        This is slow because it does not save intermediate calculations for later use.
        """
        Theo1=self.PrepareData(np.array([XYZ1]))
        Theo2=self.PrepareData(np.array([XYZ2]))
        return self.GetFastMultiDistance(Theo1,Theo2,0)
    
    def GetMultiDistance(self,XYZList,XYZ2):
        """Calculate the distance from each conformation in XYZList to XYZ2.

        Notes:
        FastMultiDistance is faster when performing many consecutive RMSDs (e.g. clustering).
        """
        TheoMulti=self.PrepareData(XYZList)
        TheoSingle=self.PrepareData(np.array([XYZ2]))
        return self.GetFastMultiDistance(TheoSingle,TheoMulti,0)
    
RMSD=RMSDMetric()
#We make an instance of our desired RMSD calculator for use in code that imports the DistanceMetric module.
