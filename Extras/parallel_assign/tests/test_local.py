import numpy as np
import numpy.testing as npt
from parallel_assign.local import partition, setup_containers, save
from nose.tools import raises
import tempfile
import tables
import IPython as ip
import os
import glob

def test_partition_0():
    project = {'TrajLengths': [2,5]}
    chunk_size = 3
        
    got = partition(project, chunk_size)
    
    
    correct = [[(0, 0, 2), (1, 0, 1)],
               [(1, 1, 4)],
               [(1, 4,5)]]

    assert [e.canonical() for e in got] == correct
    assert sum(len(e) for e in got) == sum(project['TrajLengths'])

def test_partition_1():
    project = {'TrajLengths': [2,1,10]}
    chunk_size = 4

    got = partition(project, chunk_size)
    correct = [[(0,0,2), (1,0,1), (2,0,1)],
               [(2,1,5)],
               [(2,5,9)],
               [(2,9,10)]]

    assert [e.canonical() for e in got] == correct
    assert sum(len(e) for e in got) == sum(project['TrajLengths'])

@raises(ValueError)
def test_partition_2():
    project = {'TrajLengths': [1,1,-1]}
    chunk_size = 1
    partition(project, chunk_size)

@raises(TypeError)
def test_partition_3():
    project = {'TrajLengths': [1, 1.5, 1]}
    chunk_size = 1
    partition(project, chunk_size)

@raises(ValueError)
def test_partition_3():
    project = {'TrajLengths': [1, 0]}
    chunk_size = 1
    partition(project, chunk_size)

def test_partition_4():
    project = {'TrajLengths': [1]}
    chunk_size = 11
    got = partition(project, chunk_size)
    correct = [[(0,0,1)]]
    assert [e.canonical() for e in got] == correct
    




class test_containers():
    def setup(self):
        self.d = tempfile.mkdtemp()
        project = {'TrajLengths': [9,10], 'NumTrajs':2}
        self.vtrajs = partition(project, 3)
        self.fa, self.fd = setup_containers(self.d, project, self.vtrajs)

    def test_0(self):
        assert isinstance(self.fa, tables.file.File)
        assert isinstance(self.fd, tables.file.File)
        assert self.fa.root.Data.shape == (2,10)
        assert self.fd.root.Data.shape == (2,10)
        assert self.fa.root.hashes.shape == (len(self.vtrajs), )
        assert self.fd.root.hashes.shape == (len(self.vtrajs), )
        assert self.fa.root.completed_vtrajs.shape == (len(self.vtrajs), )
        assert self.fd.root.completed_vtrajs.shape == (len(self.vtrajs), )

    def test_1(self):
        vtraj = self.vtrajs[2]
        assert len(vtraj) == 3
        
        assignments = 1234 * np.ones(len(vtraj))
        distances = np.random.randn(len(vtraj)).astype(np.float32)
        save(self.fa, self.fd, assignments, distances, vtraj)
        
        AData = -1 * np.ones((2,10))
        AData[0,6:9] = 1234
        
        DData = -1 * np.ones((2,10), dtype=np.float32)
        DData[0,6:9] = distances

        npt.assert_equal(self.fa.root.Data, AData)
        npt.assert_equal(self.fd.root.Data, DData)

    def teardown(self):
        self.fa.close()
        self.fd.close()
        for e in glob.glob(os.path.join(self.d, '*')):
            os.unlink(e)
        os.rmdir(self.d)
