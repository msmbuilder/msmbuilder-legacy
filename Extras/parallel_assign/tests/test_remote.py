import os
import numpy as np
import numpy.testing as npt
import IPython as ip

from msmbuilder import Trajectory, Project
from msmbuilder import metrics
from parallel_assign import remote
from parallel_assign.remote import assign
from parallel_assign.local import partition
from parallel_assign.vtraj import VTraj
from common import fixtures_dir


class test_assign():
    def setup(self):
        self.metric = metrics.Dihedral()
        self.pdb_fn = os.path.join(fixtures_dir(), 'native.pdb')
        self.trj_fn = os.path.join(fixtures_dir(), 'trj0.lh5')
        self.project = Project({'NumTrajs': 1, 'TrajLengths': [501], 'TrajFileBaseName': 'trj', 'TrajFileType': '.lh5',
                           'ConfFilename': self.pdb_fn,
                           'TrajFilePath': fixtures_dir()})
        self.vtraj = partition(self.project, chunk_size=501)[0]

    def test_0(self):
        assert os.path.exists(self.pdb_fn), 'file for testing not found'
        assert os.path.exists(self.trj_fn), 'file for testing not found'
    
    def test_1(self):
        # assigning some confs to themselves
        a,d,vtraj = assign(self.vtraj, self.trj_fn, self.metric)
        npt.assert_array_equal(a, np.arange(501))
        npt.assert_array_almost_equal(d, np.zeros(501), decimal=3)
        assert vtraj == self.vtraj
    
    def test_2(self):
        # reset the global
        remote.PREPARED=False
        
        # get a smaller vtraj, and just assign it to only the pDB
        vtraj = partition(self.project, chunk_size=10)[1]
        a,d,vtraj = assign(vtraj, self.pdb_fn, self.metric)

        # these are the right RMSD distances
        #correct_d = np. array([ 0.07839765,  0.07229914,  0.1135717 ,  0.14044274,  0.1121752 , 0.10593121,  0.08611701,  0.08802523,  0.08841465,  0.08553738], dtype=np.float32)
        correct_d = np.array([ 0.26932446,  0.53129266,  0.64795935,  1.56435365,  1.05962805,
                               0.60572095,  0.47062515,  0.5758602 ,  0.24565975,  0.69161412], dtype=np.float32)
        npt.assert_array_almost_equal(d, correct_d)
        npt.assert_array_equal(a, np.zeros(10))
