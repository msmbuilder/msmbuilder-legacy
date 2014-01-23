# run tests on the project class

import os

from msmbuilder.project import FahProjectBuilder
from msmbuilder import Project
from msmbuilder.project import reference_data

import mdtraj as md

import numpy as np
import numpy.testing as npt
import tempfile, shutil

from msmbuilder.testing import *

def test_project_1():
    'ensure that the counting of errors works right'
    records = {'conf_filename': None,
               'traj_lengths': [0,0,0],
               'traj_errors': [None, 1, None],
               'traj_paths': ['t0', 't1', 't2'],
               'traj_converted_from': [None, None, None]}
    proj = Project(records, validate=False)

    eq_(proj.n_trajs, 2)
    eq_(os.path.basename(proj.traj_filename(0)), 't0')

    # since t1 should be skipped
    eq_(os.path.basename(proj.traj_filename(1)), 't2')

@raises(ValueError)
def test_project_2():
    'inconsistent lengths should be detected'
    records = {'conf_filename': None,
               'traj_lengths': [0,0], # this is one too short
               'traj_errors': [None, None, None],
               'traj_paths': ['t0', 't1', 't2'],
               'traj_converted_from': [None, None, None]}
    proj = Project(records, validate=False)


def test_FahProjectBuilder_new1():
    cd = os.getcwd()
    
    native_filename = get("native.pdb", just_filename=True)
    frames_per_gen = 10
    
    traj = md.load(native_filename)
    
    fah_path = tempfile.mkdtemp()
    msmb_path = tempfile.mkdtemp()
    
    run_clone_gen = {(0, 0):5, (0, 1):6, (0, 2):7, (1, 0):20}
    reference_traj_lengths = np.array([5, 6, 7, 20]) * frames_per_gen

    ref = reference_data.FAHReferenceData(traj, fah_path, run_clone_gen, frames_per_gen)
    
    os.chdir(msmb_path)
    
    pb = FahProjectBuilder(fah_path, '.xtc', native_filename)
    project = pb.get_project()
    
    eq(project.conf_filename, native_filename)
    eq(project.traj_lengths, reference_traj_lengths)
    
    os.chdir(cd)
    shutil.rmtree(fah_path)
    shutil.rmtree(msmb_path)


def test_FahProjectBuilder_subset():
    cd = os.getcwd()
    
    native_filename = get("native.pdb", just_filename=True)
    frames_per_gen = 10
    
    traj = md.load(native_filename)
    
    fah_path = tempfile.mkdtemp()
    msmb_path = tempfile.mkdtemp()
    
    run_clone_gen = {(0, 0):5, (0, 1):6, (0, 2):7, (1, 0):20}
    reference_traj_lengths = np.array([5, 6, 7, 20]) * frames_per_gen
    
    atom_indices = np.arange(5)

    ref = reference_data.FAHReferenceData(traj, fah_path, run_clone_gen, frames_per_gen)
    
    os.chdir(msmb_path)
    
    new_native_filename = msmb_path + "/" + os.path.split(native_filename)[1]
    shutil.copy(native_filename, new_native_filename)  # Necessary on travis because we lack write permission in native.pdb directory
    
    pb = FahProjectBuilder(fah_path, '.xtc', new_native_filename, atom_indices=atom_indices)
    project = pb.get_project()
    
    new_traj = project.load_conf()
    
    eq(new_traj.n_atoms, 5)
    eq(project.traj_lengths, reference_traj_lengths)
    
    os.chdir(cd)
    shutil.rmtree(fah_path)
    shutil.rmtree(msmb_path)

