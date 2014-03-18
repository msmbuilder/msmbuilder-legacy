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

"""MSMBuilder: Markov State Models from Molecular Dynamics Simulations.

Authors:
Kyle A. Beauchamp
Gregory R. Bowman
Thomas J. Lane
Lutz Maibaum
Vijay Pande
Copyright 2011 Stanford University
"""
from __future__ import print_function, division, absolute_import


def _setup_logging():
    """Helper function to set up logger imports without polluting namespace."""
    import sys
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s', datefmt="%H:%M:%S")
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.propagate = False

_setup_logging()


# list of all the modules (files) that are part of msmbuilder
__all__ = ["metrics", "msm_analysis", "MSMLib", "clustering", "project", "reduce"]

from msmbuilder import metrics
from msmbuilder import clustering
from msmbuilder import msm_analysis
from msmbuilder import MSMLib
from msmbuilder import project
from msmbuilder import reduce
from msmbuilder.project import Project
