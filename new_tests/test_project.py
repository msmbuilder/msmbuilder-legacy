# run tests on the project class

import os

from msmbuilder.project import FahProjectBuilder
from msmbuilder import Project

import numpy.testing as npt
from nose.tools import ok_, eq_
import tempfile, shutil
from common import reference_dir, skip


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

@npt.raises(ValueError)
def test_project_2():
    'inconsistent lengths should be detected'
    records = {'conf_filename': None,
               'traj_lengths': [0,0], # this is one too short
               'traj_errors': [None, None, None],
               'traj_paths': ['t0', 't1', 't2'],
               'traj_converted_from': [None, None, None]}
    proj = Project(records, validate=False)


def test_FahProjectBuilder():
    cd = os.getcwd()
    td = tempfile.mkdtemp()
    os.chdir(td)

    traj_dir = os.path.join(reference_dir(), 
        "project_reference/project.builder/fah_style_data")
    conf_filename = os.path.join(traj_dir, 'native.pdb')
    
    pb = FahProjectBuilder(traj_dir, '.xtc', conf_filename)
    project = pb.get_project()
    eq_(project.n_trajs, 4)
    npt.assert_array_equal(project.traj_lengths, [1001, 1001, 501, 1001])
    os.chdir(cd)    
    shutil.rmtree(td)

    
    
if __name__ == '__main__':
    test_FahProjectBuilder()
