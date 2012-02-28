#!/usr/bin/python
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

import sys
import numpy

import ArgLib

from msmbuilder import Project

def run(project, clusters, stride, atomindices, globKM, locKM, output, rmsdcutoff):
    ArgLib.CheckPath(output)
    NumFrames=project["TrajLengths"].sum()/float(stride)
    SizeInBytes=2*4*NumFrames*3*project.Conf.GetNumberOfAtoms()
    print "\nNOTE: This clustering job may take up to %d MB of memory.\n" % (SizeInBytes/(1000000.))
    SkipKCenters=False
    if rmsdcutoff==0.:
        SkipKCenters=True
    Gens=project.ClusterProject(clusters,Stride=stride,GlobalKMedoidIterations=globKM,
                                LocalKMedoidIterations=locKM,AtomIndices=atomindices,RMSDCutoff=rmsdcutoff,SkipKCenters=SkipKCenters)

    Gens.SaveToLHDF(output)

    print "Wrote:", output
    return


if __name__ == "__main__":
    print """
\nClusters data based on RMSD. Implements k-centers algorithm, followed by
hybrid k-medoids to clean up data.

Output: Generators in a Trajectory format (Gens.lh5).\n"""

    arglist=["projectfn", "clusters", "stride", "atomindices",
             "globalkmediods", "localkmediods", "rmsdcutoff","outdir"]
    options=ArgLib.parse(arglist)
    print sys.argv

    P1=Project.Project.LoadFromHDF(options.projectfn)
    AInd=numpy.loadtxt(options.atomindices,int)
    output=options.outdir+'/Gens.lh5'
    stride=int(options.stride)
    rmsdcutoff=float(options.rmsdcutoff)

    run(P1, int(options.clusters), stride, AInd, int(options.globalkmediods),
        int(options.localkmediods), output,rmsdcutoff)
