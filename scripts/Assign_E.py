#!/usr/bin/env python
import sys, os
import numpy as np

from Emsmbuilder.scripts import ArgLib
from Emsmbuilder.Trajectory import Trajectory
from Emsmbuilder.Project import Project
from Emsmbuilder.assigning import simple_assign
from Emsmbuilder.metrics import RMSD

def main():
    print """\nAssigns data to generators that was not originally used in the
    clustering.\n
    Output:
    -- Assignments.h5: a matrix of assignments where each row is a vector
    corresponding to a data trajectory. The values of this vector are the cluster
    assignments.
    -- Assignments.h5.RMSD: Gives the RMSD from the assigned frame to its Generator.
    """    
    arglist=["projectfn", "generators", "atomindices", "outdir"]
    options=ArgLib.parse(arglist)
    print sys.argv

    proj = Project.LoadFromHDF(options.projectfn)
    
    metric = RMSD(np.loadtxt(options.atomindices, np.int))
    gens = Trajectory.LoadTrajectoryFile(options.generators, Conf=proj.Conf)

    assignments_path = os.path.join(options.outdir, "Assignments.h5")
    distances_path = os.path.join(options.outdir, "Assignments.h5.RMSD")


    # this prints them to disk
    all_asgn, all_dist = simple_assign(metric, proj, gens, assignments_path, distances_path)

    print 'All Done!'

if __name__ == '__main__':
    main()
