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


ATOMIC_RADII = {'H': 0.120, 'N': 0.155, 'NA': 0.227,
                'CU': 0.140, 'CL': 0.175, 'C': 0.170,
                'O': 0.152, 'I': 0.198, 'P': 0.180,
                'B': 0.185, 'BR': 0.185, 'S': 0.180,
                'SE': 0.190, 'F': 0.147, 'FE': 0.180,
                'K':  0.275, 'MN': 0.173, 'MG': 0.173,
                'ZN': 0.139, 'HG': 0.18, 'XE': 0.18,
                'AU': 0.18, 'LI': 0.18, '.': 0.18 }

def annoate_traj_with_radii(traj):
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
        if name in ATOMIC_RADII:
            traj['Radius'][i] = ATOMIC_RADII[name]
        else:
            traj['Radius'][i] = ATOMIC_RADII[name[0]]

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
        annoate_traj_with_radii(traj)
    annoate_traj_with_radii(traj)
    
    xyzlist = traj['XYZList']
    atom_radii = (traj['Radius'] + probe_radius).astype(np.float32)
    results = np.zeros((xyzlist.shape[0], xyzlist.shape[1]), dtype=np.float32)
    
    
    if not xyzlist.dtype == np.float32:
        raise TypeError('xyzlist must be float32, you supplied %s' % xyzlist.dtype)
    if not xyzlist.flags.c_contiguous:
        raise TypeError('xyzlist must be c-contiguous')
    if not atom_radii.dtype == np.float32:
        raise TypeError('atom_radii must be float32, you supplied %s' % atom_radii.dtype)
    if not n_sphere_points == int(n_sphere_points):
        raise TypeError('atom_radii must be int')
    
    _asa.asa_traj(xyzlist, atom_radii, n_sphere_points, results)

    return results
    
    
    