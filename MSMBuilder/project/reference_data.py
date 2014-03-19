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

from __future__ import print_function, absolute_import, division

import os
import numpy as np

import mdtraj as md
from mdtraj.utils.six import iteritems
import logging

logger = logging.getLogger(__name__)


def randomize_coordinates(template_traj, traj_len):
    """Randomize the coordinates in an input trajectory.

    Parameters
    ----------
    template_traj : mdtraj.Trajectory
        template_traj will be the template for randomly generated coordinate data.
        template_traj WILL BE MODIFIED inplace.
    traj_len : int
        This is the length of each trajectory (generator) to be generated.


    """
    template_traj.xyz = np.random.normal(size=(traj_len, template_traj.n_atoms, 3))
    template_traj.time = np.arange(traj_len)


class FAHReferenceData(object):

    """Generate a test case for FAH project building.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        traj will be the template for randomly generated coordinate data.
        traj WILL BE MODIFIED inplace.
    path : path where generated FAH-style project will be placed
    run_clone_gen : dict 
        "run_clone_gen[run, clone] = gen" means that `gen` trajectories
        will be generated for that run, clone pair.
    traj_len : int
        This is the length of each trajectory (generator) to be generated.

    Notes
    -----

    This will generate a directory with the following structure
    [%s/RUN%d/CLONE%d/frame%d.xtc % (path, run, clone, k)]
    where (run, clone) are the keys of run_clone_gen and k is in range(run_clone_gen[run, clone].

    """

    def __init__(self, traj, path, run_clone_gen, traj_len):

        try:
            os.mkdir(path + "/")
        except OSError:
            pass

        for (run, clone), num_gens in iteritems(run_clone_gen):

            try:
                os.mkdir(path + "/RUN%d/" % run)
            except OSError:
                pass

            os.mkdir(path + "/RUN%d/CLONE%d" % (run, clone))
            for gen in range(num_gens):
                randomize_coordinates(traj, traj_len)
                traj.save(path + "/RUN%d/CLONE%d/frame%d.xtc" % (run, clone, gen))
