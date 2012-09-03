#!/home/schwancr/epd-7.1-1-rh5-x86_64/bin/python -u
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


import os, sys
import numpy as np
from schwancrtools import ArgLib_E as ArgLib
from msmbuilder import Project, Trajectory
from schwancrtools.assigning import assign_with_checkpoint, streaming_assign_with_checkpoint
from schwancrtools.metrics_PCA import RedDimPNorm
from msmbuilder import Serializer

def main():
    print """
    \nClusters data based on RMSD. Implements k-centers algorithm, followed by
    hybrid k-medoids to clean up data.

    Output: Generators in a Trajectory format (Gens.lh5).\n

    To use other distance metrics or clustering algorithms, modify this script
    or see the examples
    """

    arglist=["projectfn","outdir","generators"]
    new_arglist=[('checkpoint','--cp','--checkpoint','Save the assignments/distances every <checkpoint> trajectories. Default: 1',1)]
    
    options, metric = ArgLib.parse(arglist,new_arglist=new_arglist,metric_parsers=True)
    print sys.argv

    # parse command line options    
    proj = Project.LoadFromHDF(options.projectfn)
    assignments_path=os.path.join( options.outdir, 'Assignments.h5')
    distances_path = os.path.join( options.outdir, 'Assignments.h5.%s' % options.metric )
    checkpoint = int( options.checkpoint )
    gens = Trajectory.LoadTrajectoryFile( options.generators )

    streaming_assign_with_checkpoint(metric, proj, gens, assignments_path, distances_path, checkpoint=checkpoint, chunk_size=100000 )
    print "Done."
if __name__ == '__main__':
    main()
    






