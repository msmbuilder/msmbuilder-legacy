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
Tests all the wrapper scripts. One test per wrapper. This should provide a
general test for the entire software package, since the wrapper scripts
sit at the very highest level and call nearly all the important functionality
underneath.

There exists some test to directly test other important parts of the software.
These have been added more piecemeal over time as needed.

Tests are deemed successful by comparing to a previously generated 'reference'
MSM. This MSM has been thoroughly checked out, so if the given output is
consistent within some tolerance, it is deemed ok.

-- TJL <tjlane@stanford.edu>, August 2011
"""

### Global Imports ###
import sys
import os
import shutil
import tempfile
import time
import unittest

import scipy
import scipy.io

import numpy as np
import numpy.testing

from msmbuilder import Project
from msmbuilder import Trajectory
from msmbuilder import Conformation
from msmbuilder import Serializer

### Local Imports ###
from msmbuilder.Scripts import Assign
from msmbuilder.Scripts import BuildMSM
from msmbuilder.Scripts import CalculateClusterRadii
from msmbuilder.Scripts import CalculateImpliedTimescales
from msmbuilder.Scripts import CalculateRMSD
from msmbuilder.Scripts import CalculateProjectRMSD
from msmbuilder.Scripts import Cluster
from msmbuilder.Scripts import ConvertDataToHDF
from msmbuilder.Scripts import CreateAtomIndices
from msmbuilder.Scripts import GetRandomConfs
from msmbuilder.Scripts import PCCA
from msmbuilder.Scripts import SavePDBs

from ReferenceParameters import *
try:
    os.mkdir(WorkingDir)
except:
    pass


class TestWrappers(unittest.TestCase):

    def assert_trajectories_equal(self, t1, t2):
        """ asserts that two MSMBuilder trajectories are equivalent """
        numpy.testing.assert_array_equal( t1["AtomID"], t2["AtomID"] )
        numpy.testing.assert_array_equal( t1["AtomNames"], t2["AtomNames"] )
        numpy.testing.assert_array_equal( t1["ChainID"], t2["ChainID"] )
        self.assertEqual( t1["IndexList"], t2["IndexList"] )
        numpy.testing.assert_array_equal( t1["ResidueID"], t2["ResidueID"] )
        numpy.testing.assert_array_equal( t1["ResidueNames"], t2["ResidueNames"] )
        numpy.testing.assert_array_almost_equal( t1["XYZList"], t2["XYZList"] )

    def test_a_ConvertDataToHDF(self):
        os.chdir(WorkingDir)
        shutil.copy(PDBFn,"./")
        ConvertDataToHDF.run(ProjectFn, PDBFn, TutorialDir+"/XTC", "file", 0, 0, 1,1000000)
        P1 = Project.Project.LoadFromHDF(ProjectFn)
        
        os.chdir(ReferenceDir)
        r_P1 = Project.Project.LoadFromHDF("ProjectInfo.h5")
        os.chdir(WorkingDir)
        
        #self.assertEqual( P1['ConfFilename'], r_P1['ConfFilename'] )
        self.assertEqual( P1['NumTrajs'], r_P1['NumTrajs'] )
        self.assertEqual( P1['TrajFileBaseName'], r_P1['TrajFileBaseName'] )

        """The following assert removed by KAB 12-12-11 because it was broken by
        recent changes to the path conventions in Project files.
        self.assertEqual( P1['TrajFilePath'], r_P1['TrajFilePath'] )
        """
        
        self.assertEqual( P1['TrajFileType'], r_P1['TrajFileType'] )
        numpy.testing.assert_array_equal( P1['TrajLengths'], r_P1['TrajLengths'] )

    def test_b_CreateAtomIndices(self):
        output = "AtomIndices.dat"
        CreateAtomIndices.run(PDBFn, 'minimal', output)
        AInd=np.loadtxt("AtomIndices.dat", int)
        r_AInd=np.loadtxt(ReferenceDir + "/AtomIndices.dat", int)
        numpy.testing.assert_array_equal(AInd, r_AInd)

    def test_c_Cluster(self):
        # We need to be sure to skip the stochastic k-mediods
        P1 = Project.Project.LoadFromHDF(ProjectFn)
        AInd = np.loadtxt("AtomIndices.dat", int)
        Cluster.run(P1, NumClusters, Stride, AInd, GlobalKMedoids, LocalKMedoids, GensPath, RMSDCutoff)
        G   = Trajectory.Trajectory.LoadTrajectoryFile(GensPath)
        r_G = Trajectory.Trajectory.LoadTrajectoryFile(ReferenceDir +'/'+ GensPath)
        self.assert_trajectories_equal(G, r_G)

    def test_d_Assign(self):
        P1 = Project.Project.LoadFromHDF(ProjectFn)
        AInd = np.loadtxt("AtomIndices.dat", int)
        Generators = Trajectory.Trajectory.LoadTrajectoryFile(GensPath)
        Assign.run(P1, AInd, Generators, "./Data/", "./Data/chkpt.h5")
        
        Assignments       = Serializer.LoadData("Data/Assignments.h5")
        AssignmentsRMSD   = Serializer.LoadData("Data/Assignments.h5.RMSD")
        
        r_Assignments     = Serializer.LoadData(ReferenceDir +"/Data/Assignments.h5")
        r_AssignmentsRMSD = Serializer.LoadData(ReferenceDir +"/Data/Assignments.h5.RMSD")

        numpy.testing.assert_array_equal( Assignments, r_Assignments )
        numpy.testing.assert_array_equal( AssignmentsRMSD, r_AssignmentsRMSD )

    def test_e_BuildMSM(self):
        Assignments = Serializer.LoadData("Data/Assignments.h5")
        BuildMSM.run(Lagtime, Assignments, Symmetrize="MLE")
        # Test mapping
        m   = np.loadtxt("Data/Mapping.dat")
        r_m = np.loadtxt(ReferenceDir +"/Data/Mapping.dat")
        numpy.testing.assert_array_almost_equal(m, r_m, err_msg="Mapping.dat incorrect")

        # Test populations
        p   = np.loadtxt("Data/Populations.dat")
        r_p = np.loadtxt(ReferenceDir +"/Data/Populations.dat")
        numpy.testing.assert_array_almost_equal(p, r_p, err_msg="Populations.dat incorrect")

        # Test counts matrix (unsymmetrized)
        uC   = scipy.io.mmread("Data/tCounts.UnSym.mtx").tocsr()
        r_uC = scipy.io.mmread(ReferenceDir +"/Data/tCounts.UnSym.mtx").tocsr()
        D=(uC-r_uC).data
        Z=0.*D#we compare the data entries of the sparse matrix
        numpy.testing.assert_array_almost_equal(D,Z, err_msg="Mapping.dat incorrect")

        # Test counts matrix
        C   = scipy.io.mmread("Data/tCounts.mtx")
        r_C = scipy.io.mmread(ReferenceDir +"/Data/tCounts.mtx")
        D=(C-r_C).data
        Z=0.*D
        numpy.testing.assert_array_almost_equal(D,Z, err_msg="tCounts.mtx incorrect")

        # Test transition matrix
        T   = scipy.io.mmread("Data/tProb.mtx")
        r_T = scipy.io.mmread(ReferenceDir +"/Data/tProb.mtx")
        D=(T-r_T).data
        Z=0.*D
        numpy.testing.assert_array_almost_equal(D,Z, err_msg="tProb.mtx incorrect")

    def test_f_CalculateImpliedTimescales(self):

        CalculateImpliedTimescales.run(MinLagtime, MaxLagtime, LagtimeInterval, NumEigen, "Data/Assignments.Fixed.h5",Symmetrize, 1, "ImpliedTimescales.dat")
        ImpTS   = np.loadtxt("ImpliedTimescales.dat")
        r_ImpTS = np.loadtxt(ReferenceDir +"/ImpliedTimescales.dat")
        numpy.testing.assert_array_almost_equal(ImpTS,r_ImpTS)

    def test_g_GetRandomConfs(self):
        # This one is tricky since it is stochastic...
        P1 = Project.Project.LoadFromHDF(ProjectFn)
        Assignments = Serializer.LoadData("Data/Assignments.Fixed.h5")
        GetRandomConfs.run(P1, Assignments, NumRandomConformations, "2RandomConfs.lh5")
        Trajectory.Trajectory.LoadTrajectoryFile("2RandomConfs.lh5")
        # Kyle: you may have a good idea for the efficient testing of this

    def test_h_CalculateClusterRadii(self):

        args = ("Data/Assignments.h5", "Data/Assignments.h5.RMSD", MinState,MaxState)
        #Note this one RETURNS a value, not saves it to disk.
        cr=CalculateClusterRadii.run( args ) #recall that this one bundles stuff
        time.sleep(10) # we have to wait a little to get results
        cr_r = np.loadtxt(ReferenceDir +"/ClusterRadii.dat")
        numpy.testing.assert_array_almost_equal( cr, cr_r )

    def test_i_CalculateRMSD(self):
        C1   = Conformation.Conformation.LoadFromPDB(PDBFn)
        Traj = Trajectory.Trajectory.LoadTrajectoryFile("Data/Gens.lh5")
        AInd = np.loadtxt("AtomIndices.dat", int)
        CalculateRMSD.run(C1, Traj, AInd, "RMSD.dat")

        cr   = np.loadtxt("RMSD.dat")
        cr_r = np.loadtxt(ReferenceDir +"/RMSD.dat")
        numpy.testing.assert_array_almost_equal( cr, cr_r )

    def test_j_PCCA(self):

        TC = scipy.io.mmread("Data/tProb.mtx")
        A  = Serializer.LoadData("Data/Assignments.Fixed.h5")
        PCCA.run(NumMacroStates, A, TC)

        ma   = Serializer.LoadData("Data/MacroAssignments.h5")
        ma_r = Serializer.LoadData(ReferenceDir +"/Data/MacroAssignments.h5")
        numpy.testing.assert_array_almost_equal( ma, ma_r )

        mm   = np.loadtxt("Data/MacroMapping.dat")
        mm_r = np.loadtxt(ReferenceDir +"/Data/MacroMapping.dat")
        numpy.testing.assert_array_almost_equal( mm, mm_r )

    def test_k_CalculateProjectRMSD(self):
        C1 = Conformation.Conformation.LoadFromPDB(PDBFn)
        P1 = Project.Project.LoadFromHDF(ProjectFn)
        AInd=np.loadtxt("AtomIndices.dat", int)
        CalculateProjectRMSD.run(C1,P1,AInd,"RMSD.h5")
        r0=Serializer.LoadData(ReferenceDir+"/RMSD.h5")
        r1=Serializer.LoadData(WorkingDir+"/RMSD.h5")
        numpy.testing.assert_array_almost_equal( r0,r1, err_msg="Error: Project RMSDs disagree!")
        
    def test_z_Cleanup(self):
        """Are we removing all unittest files? """+str(DeleteWhenFinished)
        if DeleteWhenFinished:
            shutil.rmtree(WorkingDir)

if __name__ == "__main__":
    unittest.main()
