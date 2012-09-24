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
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, 111-1307  USA

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
import pickle
import unittest

import scipy
import scipy.io

import numpy as np
from nose.tools import eq_, ok_
import numpy.testing as npt

from msmbuilder import Project
from msmbuilder import Trajectory
from msmbuilder import Conformation
from msmbuilder import io
from msmbuilder import MSMLib
from msmbuilder import tpt

### Local Imports ###
#from msmbuilder.scripts import Assign
from msmbuilder.scripts import BuildMSM
from msmbuilder.scripts import CalculateClusterRadii
from msmbuilder.scripts import CalculateImpliedTimescales
from msmbuilder.scripts import CalculateRMSD
from msmbuilder.scripts import CalculateProjectRMSD
#from msmbuilder.scripts import Cluster
from msmbuilder.scripts import ConvertDataToHDF
from msmbuilder.scripts import CreateAtomIndices
from msmbuilder.scripts import GetRandomConfs
from msmbuilder.scripts import PCCA
from msmbuilder.scripts import SavePDBs
from msmbuilder.scripts import DoTPT
from msmbuilder.scripts import FindPaths

ORIGINAL_DIRECTORY = os.getcwd()
from ReferenceParameters import *
try:
    os.mkdir(WorkingDir)
except:
    pass


class TestWrappers(unittest.TestCase):

    def assert_trajectories_equal(self, t1, t2):
        """ asserts that two MSMBuilder trajectories are equivalent """
        npt.assert_array_equal(t1["AtomID"], t2["AtomID"])
        npt.assert_array_equal(t1["AtomNames"], t2["AtomNames"])
        npt.assert_array_equal(t1["ChainID"], t2["ChainID"])
        eq_(t1["IndexList"], t2["IndexList"])
        npt.assert_array_equal(t1["ResidueID"], t2["ResidueID"])
        npt.assert_array_equal(t1["ResidueNames"], t2["ResidueNames"])
        npt.assert_array_almost_equal(t1["XYZList"], t2["XYZList"])

    def test_a_ConvertDataToHDF(self):
        os.chdir(WorkingDir)
        shutil.copy(PDBFn,"./")
                    #def run(projectfn, PDBfn, InputDir, source, mingen, stride, rmsd_cutoff,  parallel='None'):
        ConvertDataToHDF.run(ProjectFn, PDBFn, TutorialDir+"/XTC", "file", 0, 1, None)
        P1 = Project.load_from(ProjectFn)
        
        r_P1 = Project.load_from(os.path.abspath(os.path.join('..', ReferenceDir, ProjectFn)))
        
        eq_(P1.n_trajs, r_P1.n_trajs)
        npt.assert_equal(P1.traj_lengths, r_P1.traj_lengths)
        eq_(os.path.basename(P1.traj_filename(0)), os.path.basename(r_P1.traj_filename(0)))
        
    def test_b_CreateAtomIndices(self):
        AInd = CreateAtomIndices.run(PDBFn, 'minimal')
        np.savetxt("AtomIndices.dat", AInd, "%d")
        r_AInd=np.loadtxt(ReferenceDir + "/AtomIndices.dat", int)
        npt.assert_array_equal(AInd, r_AInd)

    def test_ba_tICA_train(self):
        cmd = "tICA_train.py -P 1 -d 10 -p {project} -s {stride} dihedral -a phi/psi".format(project=ProjectFn, stride=Stride )
        print cmd
        
        os.system(cmd)
        
        tICA = Serializer.LoadFromHDF( 'tICAData.h5' )

        r_tICA = Serializer.LoadFromHDF( tICADataFn )

        numpy.testing.assert_array_almost_equal( tICA['vecs'], r_tICA['vecs'] )
        numpy.testing.assert_array_almost_equal( tICA['vals'], r_tICA['vals'] )

    def test_c_Cluster(self):
        # We need to be sure to skip the stochastic k-mediods
        cmd = "Cluster.py -p {project} -s {stride} rmsd -a {atomindices} kcenters -d {rmsdcutoff}".format(project=ProjectFn, stride=Stride, atomindices="AtomIndices.dat", rmsdcutoff=RMSDCutoff)
        print cmd

        os.system(cmd)
        
        try:
            os.remove(os.path.join(WorkingDir, 'Data', 'Assignments.h5'))
            os.remove(os.path.join(WorkingDir, 'Data', 'Assignments.h5.distances'))
        except:
            pass

        
        G   = Trajectory.load_trajectory_file(GensPath)
        r_G = Trajectory.load_trajectory_file(ReferenceDir +'/'+ GensPath)
        self.assert_trajectories_equal(G, r_G)

    def test_d_Assign(self):
        cmd = "Assign.py -p %s -g %s -o %s rmsd -a %s" % (ProjectFn, GensPath, "./Data", "AtomIndices.dat")
        os.system(cmd)
        
        Assignments       = io.loadh("./Data/Assignments.h5", 'arr_0')
        AssignmentsRMSD   = io.loadh("./Data/Assignments.h5.distances", 'arr_0')
        
        r_Assignments     = io.loadh(ReferenceDir +"/Data/Assignments.h5", 'Data')
        r_AssignmentsRMSD = io.loadh(ReferenceDir +"/Data/Assignments.h5.RMSD", 'Data')
        
        npt.assert_array_equal(Assignments, r_Assignments)
        npt.assert_array_equal(AssignmentsRMSD, r_AssignmentsRMSD)
        
    
    def test_e_BuildMSM(self):
        Assignments = io.loadh("Data/Assignments.h5", 'arr_0')
        BuildMSM.run(Lagtime, Assignments, Symmetrize="MLE")
        # Test mapping
        m   = np.loadtxt("Data/Mapping.dat")
        r_m = np.loadtxt(ReferenceDir +"/Data/Mapping.dat")
        npt.assert_array_almost_equal(m, r_m, err_msg="Mapping.dat incorrect")

        # Test populations
        p   = np.loadtxt("Data/Populations.dat")
        r_p = np.loadtxt(ReferenceDir +"/Data/Populations.dat")
        npt.assert_array_almost_equal(p, r_p, err_msg="Populations.dat incorrect")

        # Test counts matrix
        C   = scipy.io.mmread("Data/tCounts.mtx")
        r_C = scipy.io.mmread(ReferenceDir +"/Data/tCounts.mtx")
        D=(C-r_C).data
        Z=0.*D

        D /= r_C.sum()#KAB 4-5-2012.  We want the normalized counts to agree at 7 decimals
        #normalizing makes this test no longer depend on an arbitrary scaling factor (the total number of counts)
        #the relative number of counts in the current and reference models DOES matter, however.
        
        npt.assert_array_almost_equal(D,Z, err_msg="tCounts.mtx incorrect")

        # Test transition matrix
        T   = scipy.io.mmread("Data/tProb.mtx")
        r_T = scipy.io.mmread(ReferenceDir +"/Data/tProb.mtx")
        D=(T-r_T).data
        Z=0.*D
        npt.assert_array_almost_equal(D,Z, err_msg="tProb.mtx incorrect")

    def test_f_CalculateImpliedTimescales(self):

        CalculateImpliedTimescales.run(MinLagtime, MaxLagtime, LagtimeInterval, NumEigen, "Data/Assignments.Fixed.h5",Symmetrize, 1, "ImpliedTimescales.dat")
        ImpTS   = np.loadtxt("ImpliedTimescales.dat")
        r_ImpTS = np.loadtxt(ReferenceDir +"/ImpliedTimescales.dat")
        npt.assert_array_almost_equal(ImpTS,r_ImpTS,decimal=4)

    def test_g_GetRandomConfs(self):
        P1 = Project.load_from(ProjectFn)
        Assignments = io.loadh("Data/Assignments.Fixed.h5", 'arr_0')
        
        # make a predictable stream of random numbers by seeding the RNG with 42
        random_source = np.random.RandomState(42)
        randomconfs = GetRandomConfs.run(P1, Assignments, NumRandomConformations, random_source)
        
        reference = Trajectory.load_trajectory_file(os.path.join(ReferenceDir, "2RandomConfs.lh5"))
        self.assert_trajectories_equal(reference, randomconfs)

    def test_h_CalculateClusterRadii(self):

        #args = ("Data/Assignments.h5", "Data/Assignments.h5.distances", MinState,MaxState)
        #Note this one RETURNS a value, not saves it to disk.
        cr = CalculateClusterRadii.main(io.loadh("Data/Assignments.h5", 'arr_0'),
                                        io.loadh("Data/Assignments.h5.distances", 'arr_0'))
        #recall that this one bundles stuff
        #time.sleep(10) # we have to wait a little to get results
        cr_r = np.loadtxt(ReferenceDir +"/ClusterRadii.dat")
        npt.assert_array_almost_equal(cr, cr_r)

    def test_i_CalculateRMSD(self):
        #C1   = Conformation.Conformation.load_from_pdb(PDBFn)
        #Traj = Trajectory.load_trajectory_file("Data/Gens.lh5")
        #AInd = np.loadtxt("AtomIndices.dat", int)
        #CalculateRMSD.run(C1, Traj, AInd, "RMSD.dat")
        outpath = os.path.join(WorkingDir, "RMSD_Gens.h5")
        os.system('CalculateProjectDistance.py -s %s -t %s -o %s rmsd -a %s' % (PDBFn, "Data/Gens.lh5", outpath, "AtomIndices.dat" ) )
        
        cr   = io.loadh(outpath, 'arr_0')
        cr_r = np.loadtxt(os.path.join(ReferenceDir, "RMSD.dat"))
        npt.assert_array_almost_equal(cr, cr_r)

        
    def test_j_PCCA(self):

        TC = scipy.io.mmread(os.path.join(WorkingDir,"Data", "tProb.mtx"))
        A  = io.loadh(os.path.join(WorkingDir,"Data", "Assignments.Fixed.h5"), 'arr_0')
        PCCA.run_pcca(NumMacroStates, A, TC, os.path.join(WorkingDir, 'Data'))

        mm   = np.loadtxt(os.path.join(WorkingDir, "Data", "MacroMapping.dat"),'int')
        mm_r = np.loadtxt(os.path.join(ReferenceDir, "Data", "MacroMapping.dat"),'int')

        ma   = io.loadh(os.path.join(WorkingDir, "Data", "MacroAssignments.h5"), 'arr_0')
        ma_r = io.loadh(os.path.join(ReferenceDir, "Data", "MacroAssignments.h5"), 'Data')

        num_macro = NumMacroStates
        permutation_mapping = np.zeros(num_macro,'int')
        #The order of macrostates might be different between the reference and new lumping.
        #We therefore find a permutation to match them.
        for i in range(num_macro):
            j = np.where(mm==i)[0][0]
            permutation_mapping[i] = mm_r[j]

        mm_permuted = permutation_mapping[mm]
        MSMLib.ApplyMappingToAssignments(ma,permutation_mapping)
        
        npt.assert_array_almost_equal(mm_permuted, mm_r)
        npt.assert_array_almost_equal(ma, ma_r)

    def test_k_CalculateProjectRMSD(self):
        #C1 = Conformation.load_from_pdb(PDBFn)
        #P1 = Project.load_from_hdf(ProjectFn)
        #AInd=np.loadtxt("AtomIndices.dat", int)
        #CalculateProjectRMSD.run(C1,P1,AInd,"RMSD.h5")
        outpath = os.path.join(WorkingDir, "RMSD.h5")
        os.system('CalculateProjectDistance.py -s %s -o %s -p %s rmsd -a %s' % (PDBFn, outpath, ProjectFn, "AtomIndices.dat") )
        
        
        r0 = io.loadh(ReferenceDir+"/RMSD.h5", 'Data')
        r1 = io.loadh(WorkingDir+"/RMSD.h5", 'arr_0')
        npt.assert_array_almost_equal(r0,r1, err_msg="Error: Project RMSDs disagree!")

    def test_l_CalculateProjectSASA(self):
        outpath = os.path.join(WorkingDir, "SASA.h5")
        os.system('CalculateProjectSASA.py -o %s -p %s' % (outpath, ProjectFn) )

        r0 = io.loadh(os.path.join( ReferenceDir, "SASA.h5" ), 'Data')
        r1 = io.loadh(os.path.join( WorkingDir, "SASA.h5" ), 'arr_0')
        npt.assert_array_almost_equal(r0,r1, err_msg="Error: Project SASAs disagree!")

    def test_m_DoTPT(self): 
        T = scipy.io.mmread(os.path.join(ReferenceDir, "Data", "tProb.mtx"))
        sources = [0]
        sinks = [70]
        script_out = DoTPT.run(T, sources, sinks)
        committors_ref = io.loadh(os.path.join(ReferenceDir, "transition_path_theory_reference", "committors.h5"), 'Data')
        net_flux_ref = io.loadh(os.path.join(ReferenceDir, "transition_path_theory_reference", "net_flux.h5"), 'Data')
        npt.assert_array_almost_equal(script_out[0], committors_ref)
        npt.assert_array_almost_equal(script_out[1].toarray(), net_flux_ref)

    def test_n_FindPaths(self):
        tprob = scipy.io.mmread(os.path.join(ReferenceDir, "Data", "tProb.mtx"))
        sources = [0]
        sinks = [70]
        paths, bottlenecks, fluxes = FindPaths.run(tprob, sources, sinks, 10)
        # paths are hard to test due to type issues, adding later --TJL
        bottlenecks_ref = io.loadh(os.path.join(ReferenceDir, "transition_path_theory_reference", "dijkstra_bottlenecks.h5"), 'Data')
        fluxes_ref = io.loadh(os.path.join(ReferenceDir, "transition_path_theory_reference", "dijkstra_fluxes.h5"), 'Data')
        npt.assert_array_almost_equal(bottlenecks, bottlenecks_ref)
        npt.assert_array_almost_equal(fluxes, fluxes_ref)

    def test_z_Cleanup(self):
        """Are we removing all unittest files? """+str(DeleteWhenFinished)
        if DeleteWhenFinished:
            shutil.rmtree(WorkingDir)
        os.chdir(ORIGINAL_DIRECTORY)

if __name__ == "__main__":
    unittest.main()
