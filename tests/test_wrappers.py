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
New tests for the scripts. The idea is to replace the current TestWrappers.py
with this. The improvements are basically towards maintainability, and are
documented in PR 48 at https://github.com/SimTk/msmbuilder/pull/48
"""

import numpy as np
import tempfile
import shutil
import os
import tarfile
from unittest import skipIf
import nose

import mdtraj as md
from msmbuilder import MSMLib
from msmbuilder import metrics
from msmbuilder.testing import get, eq, load

from msmbuilder.scripts import ConvertDataToHDF
from msmbuilder.scripts import CreateAtomIndices
from msmbuilder.scripts import tICA_train
from msmbuilder.scripts import Cluster
from msmbuilder.scripts import Assign
from msmbuilder.scripts import AssignHierarchical
from msmbuilder.scripts import BuildMSM
from msmbuilder.scripts import CalculateImpliedTimescales
from msmbuilder.scripts import PCCA
from msmbuilder.scripts import CalculateTPT
from msmbuilder.scripts import FindPaths
from msmbuilder.scripts import CalculateClusterRadii
from msmbuilder.scripts import CalculateMFPTs

from msmbuilder.scripts import CalculateProjectDistance


def pjoin(*args):
    return os.path.join(*args)
# CRS: is this function really necessary?

# subclassing this will get you a test where you have a temporary
# directory (self.td) that will get cleaned up when you tes finishes
# (whether it passes or fails)
class WTempdir(object):
    def setup(self):
        self.td = tempfile.mkdtemp()
        os.chdir(self.td)

    def teardown(self):
        shutil.rmtree(self.td)


class test_ConvertDataToHDF(WTempdir):
    def test(self):
        # extract xtcs to a temp dir
        xtc_fn = get('XTC.tgz', just_filename=True)
        fh = tarfile.open(xtc_fn, mode='r:gz')
        fh.extractall(self.td)
        fh.close()

        outfn = pjoin(self.td, 'ProjectInfo.yaml')
        # move to that directory
        os.chdir(self.td)
        ConvertDataToHDF.run(projectfn=outfn,
                             conf_filename=get('native.pdb', just_filename=True),
                             input_dir=pjoin(self.td, 'XTC'),
                             source='file',
                             min_length=0,
                             stride=1,
                             rmsd_cutoff=np.inf, atom_indices=None, iext=".xtc")

        eq(load(outfn), get('ProjectInfo.yaml'))


class test_ConvertDataToHDF_atomindices(WTempdir):
    def test(self):
        # extract xtcs to a temp dir
        xtc_fn = get('XTC.tgz', just_filename=True)
        fh = tarfile.open(xtc_fn, mode='r:gz')
        fh.extractall(self.td)
        fh.close()

        outfn = pjoin(self.td, 'ProjectInfo.yaml')
        # move to that directory
        os.chdir(self.td)

        atom_indices = np.arange(4)

        ConvertDataToHDF.run(projectfn=outfn,
                             conf_filename=get('native.pdb', just_filename=True),
                             input_dir=pjoin(self.td, 'XTC'),
                             source='file',
                             min_length=0,
                             stride=1,
                             rmsd_cutoff=np.inf,
                             atom_indices=atom_indices, iext=".xtc")

        project = load(outfn)
        traj = project.load_conf()
        eq(traj.n_atoms, 4)

class test_tICA_train(WTempdir):
    def test(self):

        prep_metric = metrics.Dihedral(angles='phi/psi')
        project = get('ProjectInfo.yaml')

        os.chdir(self.td)
        tICA_train.run(prep_metric, project, delta_time=10, atom_indices=None,
                       output='tICAtest.h5', min_length=0, stride=1)

        ref_tICA = get('tICA_ref_mle.h5')

        ref_vals = ref_tICA['vals']
        ref_vecs = ref_tICA['vecs']
        ref_inds = np.argsort(ref_vals)
        ref_vals = ref_vals[ref_inds]
        ref_vecs = ref_vecs[:, ref_inds]

        test_tICA = load('tICAtest.h5')

        test_vals = test_tICA['vals']
        test_vecs = test_tICA['vecs']
        test_inds = np.argsort(test_vals)
        test_vals = test_vals[test_inds]
        test_vecs = test_vecs[:, test_inds]

        eq(test_vals, ref_vals)
        eq(test_vecs, test_vecs)

def test_CreateAtomIndices():
    indices = CreateAtomIndices.run(get('native.pdb', just_filename=True),
                                   'minimal')
    eq(indices, get('AtomIndices.dat'))


class test_Cluster_kcenters(WTempdir):
    # this one tests kcenters
    def test(self):
        args, metric = Cluster.parser.parse_args([
            '-p', get('points_on_cube/ProjectInfo.yaml', just_filename=True),
            '-o', self.td,
            'rmsd', '-a', get('points_on_cube/AtomIndices.dat', just_filename=True),
            'kcenters', '-k', '4'], print_banner=False)
        Cluster.main(args, metric)

        assignments = load(pjoin(self.td, 'Assignments.h5'))["arr_0"]
        assignment_counts = np.bincount(assignments.flatten())
        eq(assignment_counts, np.array([2, 2, 2, 2]))

        distances = load(pjoin(self.td, 'Assignments.h5.distances'))["arr_0"]
        eq(distances, np.zeros((1,8)))


class test_Cluster_hybrid(WTempdir):
    # this one tests hybrid kcenters kmedoids
    def test(self):
        args, metric = Cluster.parser.parse_args([
            '-p', get('points_on_cube/ProjectInfo.yaml', just_filename=True),
            '-o', self.td,
            'rmsd', '-a', get('points_on_cube/AtomIndices.dat', just_filename=True),
            'hybrid', '-k', '4'], print_banner=False)
        Cluster.main(args, metric)


class test_Cluster_hierarchical(WTempdir):
    def test(self):
        if os.getenv('TRAVIS', None) == 'true':
            raise nose.SkipTest('Skipping test_Assign on TRAVIS')

        try:
            import fastcluster
        except ImportError:
            raise nose.SkipTest("Cannot find fastcluster, so skipping hierarchical clustering test.")

        args, metric = Cluster.parser.parse_args([
            '-p', get('ProjectInfo.yaml', just_filename=True),
            '-s', '10',
            '-o', self.td,
            'rmsd', '-a', get('AtomIndices.dat', just_filename=True),
            'hierarchical'],
            print_banner=False)
        Cluster.main(args, metric)

        eq(load(pjoin(self.td, 'ZMatrix.h5')), get('ZMatrix.h5'))


class test_Assign(WTempdir):
    def test(self):
        args, metric = Assign.parser.parse_args([
            '-p', get('ProjectInfo.yaml', just_filename=True),
            '-g', get('Gens.lh5', just_filename=True),
            '-o', self.td,
            'rmsd', '-a', get('OldAtomIndices.dat', just_filename=True)],
            print_banner=False)
        if os.getenv('TRAVIS', None) == 'true':
            raise nose.SkipTest('Skipping test_Assign on TRAVIS')

        Assign.main(args, metric)

        eq(load(pjoin(self.td, 'Assignments.h5')),
           get('assign/Assignments.h5'))
        eq(load(pjoin(self.td, 'Assignments.h5.distances')),
           get('assign/Assignments.h5.distances'), decimal=5)


class test_AssignHierarchical(WTempdir):
    def test(self):
        project = get('ProjectInfo.yaml')
        asgn = AssignHierarchical.main(k=100, d=None,
            zmatrix_fn=get('ZMatrix.h5', just_filename=True), stride=10, project=project)

        eq(asgn, get('WardAssignments.h5')['arr_0'])


class test_BuildMSM(WTempdir):
    def test(self):
        BuildMSM.run(lagtime=1, assignments=get('Assignments.h5')['arr_0'], symmetrize='MLE',
            out_dir=self.td)

        eq(load(pjoin(self.td, 'tProb.mtx')), get('tProb.mtx'), decimal=5)
        eq(load(pjoin(self.td, 'tCounts.mtx')), get('tCounts.mtx'), decimal=3)
        eq(load(pjoin(self.td, 'Mapping.dat')), get('Mapping.dat'))
        eq(load(pjoin(self.td, 'Assignments.Fixed.h5')), get('Assignments.Fixed.h5'))
        eq(load(pjoin(self.td, 'Populations.dat')), get('Populations.dat'))


def test_CalculateImpliedTimescales():
    impTimes = CalculateImpliedTimescales.run(MinLagtime=3, MaxLagtime=5,
        Interval=1, NumEigen=10, AssignmentsFn=get('Assignments.h5', just_filename=True),
        trimming=True, symmetrize='Transpose', nProc=1)

    eq(impTimes, get('ImpliedTimescales.dat'))


def test_CalculateMFPTs(mfpt_state=70):
    mfpt = CalculateMFPTs.run(get('transition_path_theory_reference/tProb.mtx'), mfpt_state)
    mfpt0 = get(pjoin("transition_path_theory_reference", "mfpt.h5"))['Data']
    eq(mfpt, mfpt0)


def test_CalculateTPT():
    T = get("transition_path_theory_reference/tProb.mtx")
    sources = [0]  # chosen arb. for ref. by TJL
    sinks = [70]  # chosen arb. for ref. by TJL
    script_out = CalculateTPT.run(T, sources, sinks)
    committors_ref = get(pjoin("transition_path_theory_reference",
                          "committors.h5"))['Data']
    net_flux_ref = get(pjoin("transition_path_theory_reference",
                        "net_flux.h5"))['Data']
    eq(script_out[0], committors_ref)
    eq(script_out[1].toarray(), net_flux_ref)


def test_CalculateClusterRadii():
    cr = CalculateClusterRadii.main(get("Assignments.h5")['arr_0'],
                                    get("Assignments.h5.distances")['arr_0'])
    cr_r = get("ClusterRadii.dat")
    eq(cr, cr_r)


def test_FindPaths():
    tprob = get("transition_path_theory_reference/tProb.mtx")
    sources = [0]
    sinks = [70]
    paths, bottlenecks, fluxes = FindPaths.run(tprob, sources, sinks, 10)
    # paths are hard to test due to type issues, adding later --TJL
    bottlenecks_ref = get(pjoin("transition_path_theory_reference",
                           "dijkstra_bottlenecks.h5"))['Data']
    fluxes_ref = get(pjoin("transition_path_theory_reference",
                      "dijkstra_fluxes.h5"))['Data']
    eq(bottlenecks, bottlenecks_ref)
    eq(fluxes, fluxes_ref)


class test_PCCA(WTempdir):

    def test(self):

        num_macro = 5

        TC = get("PCCA_ref/tProb.mtx")
        A = get("PCCA_ref/Assignments.Fixed.h5")['arr_0']
        print A

        macro_map, macro_assign = PCCA.run_pcca(num_macro, A, TC)
        r_macro_map = get("PCCA_ref/MacroMapping.dat")

        macro_map = macro_map.astype(np.int)
        r_macro_map = r_macro_map.astype(np.int)

        # The order of macrostates might be different between the reference and
        # new lumping. We therefore find a permutation to match them.
        permutation_mapping = np.zeros(macro_assign.max() + 1, 'int')
        for i in range(num_macro):
            j = np.where(macro_map == i)[0][0]
            permutation_mapping[i] = r_macro_map[j]

        macro_map_permuted = permutation_mapping[macro_map]
        MSMLib.apply_mapping_to_assignments(macro_assign, permutation_mapping)

        r_macro_assign = get("PCCA_ref/MacroAssignments.h5")['arr_0']

        eq(macro_map_permuted, r_macro_map)
        eq(macro_assign, r_macro_assign)


class test_SaveStructures(WTempdir):
    def test(self):
        from msmbuilder.scripts.SaveStructures import save

        os.chdir(self.td)

        project = get('ProjectInfo.yaml')
        assignments = get('Assignments.h5')['arr_0']
        which_states = [0, 1, 2]
        list_of_trajs = project.get_random_confs_from_states(assignments,
            which_states, num_confs=2, replacement=True,
            random=np.random.RandomState(42))

        assert isinstance(list_of_trajs, list)
        assert isinstance(list_of_trajs[0], md.Trajectory)
        eq(len(list_of_trajs), len(which_states))
        for t in list_of_trajs:
            eq(len(t), 2)

        # sep, tps, one
        save(list_of_trajs, which_states, style='sep', format='lh5', outdir=self.td)
        save(list_of_trajs, which_states, style='tps', format='lh5', outdir=self.td)
        save(list_of_trajs, which_states, style='one', format='lh5', outdir=self.td)

        names = ['State0-0.lh5', 'State0-1.lh5', 'State0.lh5', 'State1-0.lh5',
                'State1-1.lh5', 'State1.lh5', 'State2-0.lh5', 'State2-1.lh5',
                'State2.lh5']

        for name in names:
            t = md.load(pjoin(self.td, name))
            eq(t.xyz, get('save_structures/' + name).xyz)  # Just checking coordinates because atom names / bonds in reference data are incompatible with MDTraj.
