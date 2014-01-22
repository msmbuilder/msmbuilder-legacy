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

import os
import numpy as np

from msmbuilder import MSMLib
import mdtraj as md
import logging
from msmbuilder.utils import keynat

logger = logging.getLogger(__name__)

def randomize_coordinates(template_traj, traj_len):
    template_traj.xyz = np.random.normal(size=(traj_len, template_traj.n_atoms, 3))
    template_traj.time = np.arange(traj_len)
    

class FAHReferenceData(object):
    """Generate a test case for FAH project building.
    
    Notes
    -----
    
    This will generate a directory with 
    [/RUN%d/CLONE%d/frame%d.xtc % (i, j, k)]

    """
    def __init__(self, traj, path, run_clone_gen, traj_len):
        
        try:
            os.mkdir(path + "/")
        except OSError:
            pass
        
        for (run, clone), num_gens in run_clone_gen.iteritems():
        
            try:
                os.mkdir(path + "/RUN%d/" % run)
            except OSError:
                pass
        
            os.mkdir(path + "/RUN%d/CLONE%d" % (run, clone))
            for gen in range(num_gens):
                randomize_coordinates(traj, traj_len)
                traj.save(path + "/RUN%d/CLONE%d/frame%d.xtc" % (run, clone, gen))
                
