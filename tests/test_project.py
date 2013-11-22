# run tests on the project class

import os

from msmbuilder.project import FahProjectBuilder
from msmbuilder import Project

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


def test_FahProjectBuilder1():
    cd = os.getcwd()
    td = tempfile.mkdtemp()
    os.chdir(td)

    # check that we can build a new project:
    traj_dir = get("project_reference/project.builder/fah_style_data", just_filename=True)

    shutil.copytree(traj_dir, 'PROJXXXX')
    shutil.rmtree('PROJXXXX/RUN0/CLONE1')
    os.remove('PROJXXXX/RUN2/CLONE0/frame2.xtc')
    # made up project data

    pb = FahProjectBuilder('PROJXXXX', '.xtc', 'PROJXXXX/native.pdb')
    project = pb.get_project()
    project_ref = get("project_reference/project.builder/ProjectInfo.yaml")

    print project == project_ref
    assert project == project_ref

    os.chdir(cd)
    shutil.rmtree(td)


def test_FahProjectBuilder2():
    cd = os.getcwd()
    td = tempfile.mkdtemp()
    os.chdir(td)

    # check that we can build a new project:
    traj_dir = get("project_reference/project.builder/fah_style_data", just_filename=True)
    conv_traj_dir = get("project_reference/project.builder/Trajectories", just_filename=True)
    shutil.copytree(traj_dir, 'PROJXXXX')
    shutil.copytree(conv_traj_dir, 'Trajectories')
    shutil.copy2(get("project_reference/project.builder/ProjectInfo.yaml", just_filename=True), 'ProjectInfo.yaml')
    project_orig = Project.load_from('ProjectInfo.yaml')
    # made up project data

    pb = FahProjectBuilder('PROJXXXX', '.xtc', 'PROJXXXX/native.pdb', project=project_orig)
    project = pb.get_project()
    project_ref = get("project_reference/project.builder/ProjectInfo_final.yaml")

    assert project == project_ref

    os.chdir(cd)
    shutil.rmtree(td)

if __name__ == '__main__':
    test_FahProjectBuilder1()
    test_FahProjectBuilder2()
