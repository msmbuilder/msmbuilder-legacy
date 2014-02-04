#!/usr/bin/env python
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
import logging
import numpy as np
import scipy.io
from msmbuilder.tpt import calculate_committors, calculate_net_fluxes
from msmbuilder import arglib
logger = logging.getLogger('msmbuilder.scripts.CalculateTPT')

parser = arglib.ArgumentParser(description="""
    Calculates a number of kinetic transition properties of a given MSM. Returns:
    (1) committors.dat - the forward committors of the MSM (numpy savetxt)
    (3) net_flux.mtx - the net flux matrix (scipy sparse fmt)""")
parser.add_argument('tProb')
parser.add_argument('starting', help='''Vector of states in the
    starting/reactants/unfolded ensemble.''', default='U_states.dat')
parser.add_argument('ending', help='''Vector of states in the
    ending/products/folded ensemble.''', default='F_states.dat')
parser.add_argument('output_dir', default='.')


def run(TC, Uv, Fv):

    # Get committors and flux
    logger.info("Getting committors and flux...")

    Fc = calculate_committors(Uv, Fv, TC)
    logger.info("Calculated forward committors.")

    NFlux = calculate_net_fluxes(Uv, Fv, TC)
    logger.info("Calculated net flux.")

    return Fc, NFlux


if __name__ == "__main__":
    args = parser.parse_args()

    T = scipy.io.mmread(args.tProb)
    U = np.loadtxt(args.starting).astype(int)
    F = np.loadtxt(args.ending).astype(int)

    # deal with case where have single start or end state
    # TJL note: This should be done in the library now... but leaving it
    if U.shape == ():
        tmp = np.zeros(1, dtype=int)
        tmp[0] = int(U)
        U = tmp.copy()
    if F.shape == ():
        tmp = np.zeros(1, dtype=int)
        tmp[0] = int(F)
        F = tmp.copy()

    # Check output isn't taken
    output_list = ["committors.dat", "net_flux.mtx"]
    output_flist = [os.path.join(args.output_dir, f) for f in output_list]
    arglib.die_if_path_exists(output_flist)

    Fc, NFlux = run(T, U, F)

    np.savetxt(output_flist[0], Fc)
    scipy.io.mmwrite(output_flist[1], NFlux)
    logger.info("Saved output to %s", ', '.join(output_flist))
