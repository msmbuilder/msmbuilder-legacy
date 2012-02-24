'''
Created on Nov 23, 2010

@author: gbowman
'''
import numpy.testing
import os
import os.path
import tables
import tempfile
import unittest

from Emsmbuilder import Serializer
import numpy as np

Data=np.arange(10000,dtype='float')
Filename1 = "MSMB_UnitTest1.h5"
Filename2 = "MSMB_UnitTest2.h5"

class TestSerializer(unittest.TestCase):
    """unittest runs tests in alphabetical order.  So, the load function will be tested before the save function, which is useful 
    since testing the save function requires loading the result to check its correctness."""

    def setUp(self):
        """setup() is called before very test and just creates a temporary work space for reading/writing files."""
        pass

    def test0_CheckFilenames(self):
        """Make sure that temporary files do not already exist."""
        for Filename in [Filename1,Filename2]:
            if os.path.exists(Filename):
                raise Exception("Cannot Perform Unit Test: %d already exists!"%Filename)

    def test1_HDFWriting(self):
        """Write Data to an HDF5 file as a compressed CArray."""
        hdfFile = tables.File(Filename1, 'a')
        #The filter is the same used to save MSMB2 data
        hdfFile.createCArray("/", "Data", tables.Float32Atom(),Data.shape,filters=Serializer.Filter)
        hdfFile.root.Data[:]=Data[:]
        hdfFile.flush()
        hdfFile.close()
        
    def test2_LoadFromHDF(self):
        """Load HDF5 file from (0) and verify correctness."""
        TestData=Serializer.LoadData(Filename1)        
        # check correctness or raise error
        numpy.testing.assert_array_equal(TestData,Data)
            
    def test3_SaveToHDF(self):
        """Save HDF5 to disk usingSaveData."""
        Serializer.SaveData(Filename2,Data)

    def test4_LoadFromSavedHDF(self):
        """Load the results from (2) and check against known answer.

        """
        TestData=Serializer.LoadData(Filename2)
        numpy.testing.assert_array_equal(TestData,Data)

    def test5_Cleanup(self):
        """Cleaning up Serializer test files.

        """
        os.remove(Filename1)
        os.remove(Filename2)
        
if __name__ == "__main__":
    unittest.main()
