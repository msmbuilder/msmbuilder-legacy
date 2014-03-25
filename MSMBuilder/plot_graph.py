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

"""Code for visualizing networks (e.g. transition, count, or flux matrices). 
"""
from __future__ import print_function, division, absolute_import

import scipy.sparse
import numpy as np
import networkx
import sys
import re
import logging
logger = logging.getLogger(__name__)

def CreateNetwork(Matrix, EqPops, Directed=True, EdgeScale=2, PopCutoff=0.01, EdgeCutoff=0.1, ImageList=None, Labels=None):
    """Creates a NetworkX graph (or DiGraph) object.

    Inputs:
    Matrix -- a sparse matrix.
    EqPops -- state equilibrium populations.

    Keyword Arguments:
    Directed -- Is the graph directed?  Default: True
    EdgeScale -- a scale factor for edges.  Default: 2
    PopCutoff -- hide states with populations lower than this.  Default: 0.01
    EdgeCutoff -- hide edges with weights lower than this.  Default: 0.1
    ImageList -- A list of filenames for visualization on each states.  Default: None
    Labels -- A List of labels for display on each state.  Default: None

    Notes:
    The NetworkX graph can be used for simple visualization (e.g. matplotlib) or export to a GraphViz dotfile.
    You can optionally add image paths to the graph, which will be recorded in the dotfile for eventual visualization.
    """

    Matrix=Matrix.tocsr()
    print("Loaded an MSM with %d states..." % Matrix.shape[0])

    #These are the desired states.
    Ind=np.where(EqPops>PopCutoff)[0]
    if len(Ind) == 0:
        raise ValueError("Error! No nodes will be rendered. Try lowering the population cutoff (epsilon).")

    # if user specified labels use those, otherwise use Ind
    if Labels == None:
        Labels = Ind

    #Select the desired subset of Matrix
    Matrix=Matrix[Ind][:,Ind]
    EqEnergy=-np.log(EqPops[Ind])

    # invert thigns so less populated states are smaller than more populated ones
    maxEqEnergy = EqEnergy.max()
    EqEnergy = 1 + maxEqEnergy - EqEnergy # add 1 to ensure nothing gets 0 weight

    # Renormalize stuff to make it more reasonable
    frm, to, weight = scipy.sparse.find( Matrix )
    weight /= weight.max() 
    weight *= EdgeScale
    
    n=Matrix.shape[0]
    
    if Directed:
        G=networkx.from_scipy_sparse_matrix(Matrix,create_using=networkx.DiGraph())
    else:
        G=networkx.from_scipy_sparse_matrix(Matrix)
    logger.info("Rendering %d nodes...", n)
            
    # Write attributes to G
    for i in range(n):
        G.node[i]["width"]=EqEnergy[i]/3
        G.node[i]["height"]=EqEnergy[i]/3
        G.node[i]["fixedsize"] = True
        G.node[i]["label"]=Labels[i]

    for i in range(len(weight)):
        G[frm[i]][to[i]]["penwidth"]=weight[i]*2
        G[frm[i]][to[i]]["arrowsize"]=weight[i]*2

    #Save image paths if desired.
    if ImageList!=None:
        logger.info("Found %d images - attempting to include them in the .dot file", len(ImageList))
        ImageList=np.array(ImageList)
        for Image in ImageList:
            match = re.findall( '(\d+)', Image )
            if len(match) > 0:
                state = int( match[0] )
                if state in Ind:
                    Gind = int(np.where( Ind == state)[0])
                    G.node[Gind]["image"]=Image
                    logger.info("Found an image for state: %d", state)

    return(G)

def PlotNetwork(G,OutputFile="Graph.dot"):
    """Plot a graph to screen and also save output."""

    try:
        networkx.draw_graphviz(G)
    except:
        logger.error("could not plot graph to screen.  Check X / Matplotlib settings.")
        
    networkx.write_dot(G, OutputFile)
    logger.info("Wrote: %s", OutputFile)
