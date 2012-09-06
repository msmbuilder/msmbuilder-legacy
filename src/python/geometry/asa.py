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
import numpy as np
from msmbuilder import _asa

# these van der waals radii are taken from GROMACS 4.5.3
# and the file share/gromacs/top/vdwradii.dat   
ATOMIC_RADII = {'C': 0.15,  'F': 0.12,  'H': 0.04,
                'N': 0.110, 'O': 0.105, 'S': 0.16}


def annotate_traj_with_radii(traj):
    """Add the key 'Radius' to traj, with the van Der Waals radii of the atoms

    Acts in place

    Parameters
    ----------
    traj : msmbuilder.Trajectory
        A trajectory to work on
    """

    traj['Radius'] = np.zeros(len(traj['AtomNames']), dtype=np.float32)
    for i, name in enumerate(traj['AtomNames']):
        name = name.strip('0123456789 ').upper()
        first_letter = name[0]
        try:
            traj['Radius'][i] = ATOMIC_RADII[first_letter]
        except KeyError:
            raise KeyError('The atom name %s was not able to be parsers into a van der walls radius' % name)

def calculate_asa(traj, probe_radius=0.14, n_sphere_points=960):
    """
    Calculate the accessible surface area of each atom in every snapshot
    of an MD trajectory

    Parameters
    ----------
    traj : msmbuilder.Trajectory
        A trajectory to calculate on
    probe_radius : float, optional
        The radius of the probe in nm
    n_sphere_pts : int, optional
        The number of points representing the surface of each atom, higher
        values leads to more accuracy

    Returns
    -------
    areas : np.array, shape=[n_frames, n_atoms]
        The accessible surface area of each atom in every frame
    """
    if not traj.has_key("Radius"):
        annotate_traj_with_radii(traj)
    annotate_traj_with_radii(traj)
    
    xyzlist = traj['XYZList']
    atom_radii = (traj['Radius'] + probe_radius).astype(np.float32)

    return _asa.asa_trajectory(xyzlist, atom_radii, n_sphere_points)
    
    
    
