'''
Created on Nov 23, 2010 by gbowman
Expanded & Updated August 2011 by tjlane
'''

import unittest
import os
import sys

def clean():
    """do some manual clean up"""
    print """======================================================================
PERFORMING CLEANUP
----------------------------------------------------------------------"""

    cmds = [
           ]

    for cmd in cmds:
        print "EXECUTING:", cmd
        os.system(cmd)

    print """----------------------------------------------------------------------"""

# check to see if all we want to do is a cleaning
if len(sys.argv) == 2:
    if sys.argv[1] == 'clean': clean(); sys.exit(0)

#from TestMSMLib import TestMSMLib
from TestSerializer import TestSerializer
from TestWrappers import TestWrappers

# create test suite and add individual tests.
suite = unittest.TestSuite()
suite.addTest(unittest.makeSuite(TestSerializer))
suite.addTest(unittest.makeSuite(TestWrappers))

#suite.addTest(unittest.makeSuite(TestMSMLib))



# run all the tests
unittest.TextTestRunner(verbosity=2).run(suite)

clean()

