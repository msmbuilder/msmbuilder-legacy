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
"""
Calculate the solvent accessible surface area (ASA) for a single frame
in a trajectory.

Note: This code is quite slow because it requires a triply nested for loop
in python. If you need to use it for a big system or a lot of frames,
I think its possible that about ~5 hours of work writing the algorithm in
C could speed it up by >50x.
"""


from __future__ import division
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.spatial.distance import euclidean, sqeuclidean

ATOMIC_RADII = {'H': 0.120, 'N': 0.155, 'NA': 0.227,
                'CU': 0.140, 'CL': 0.175, 'C': 0.170,
                'O': 0.152, 'I': 0.198, 'P': 0.180,
                'B': 0.185, 'BR': 0.185, 'S': 0.180,
                'SE': 0.190, 'F': 0.147, 'FE': 0.180,
                'K':  0.275, 'MN': 0.173, 'MG': 0.173,
                'ZN': 0.139, 'HG': 0.18, 'XE': 0.18,
                'AU': 0.18, 'LI': 0.18, '.': 0.18 }


def calculate_asa(traj, frame_indx=0, probe_radius=0.14, n_sphere_pts=960):
    """
    Calculate the accessible surface area of each atom

    Parameters
    ----------
    traj : msmbuilder.Trajectory
        A trajectory to calculate on
    frame_indx : int
        The frame in the trajectory you wish to calculate for
    probe_radius : float
        The radius of the probe in nm
    n_sphere_pts : int, optional
        The number of points representing the surface of each atom, higher
        values leads to more accuracy

    Returns
    -------
    areas : np.array, shape=[n_atoms]
        The accessible surface area of each atom
    """
    if not traj.has_key("Radius"):
        annoate_traj_with_radii(traj)
        
    sphere_points = generate_sphere_points(n_sphere_pts)
    const = 4.0 * np.pi / n_sphere_pts
    
    xyzlist = traj['XYZList']
    radii = traj['Radius']
    n_atoms = xyzlist.shape[1]
    
    areas = np.zeros(n_atoms)
    
    # calculate the full all-to-all distance matrix
    distance_mtx = squareform(pdist(xyzlist[frame_indx, :, :]))
    
    for i in xrange(n_atoms):
        # indices of the atoms close to atom `i`
        possible = np.concatenate((np.arange(i), np.arange(i + 1, n_atoms)))
        neighbor_indices = possible[np.where(distance_mtx[possible, i] < \
                                             radii[i] + 2 * probe_radius + \
                                             radii[possible])]
        
        # a bunch of points on the surface of atom i
        atom_centered_pts = xyzlist[frame_indx, i] + \
            sphere_points * (probe_radius + radii[i])
        
        n_accessible_point, j_closest_neighbor = 0, 0
        
        # check if each of these points is accessible
        for l in xrange(atom_centered_pts.shape[0]):
            test_point = atom_centered_pts[l]
            is_accessible = True
            
            cycled_indices = range(j_closest_neighbor, len(neighbor_indices))
            cycled_indices.extend(range(j_closest_neighbor))

            for j in cycled_indices:
                r = radii[neighbor_indices[j]] + probe_radius
                diff_sq = sqeuclidean(test_point,
                                      xyzlist[frame_indx, neighbor_indices[j]])
                if diff_sq < r**2:
                    j_closest_neighbor = j
                    is_accessible = False
                    break

            if is_accessible:
                n_accessible_point += 1
        
        areas[i] = const * n_accessible_point * (probe_radius + radii[i])**2

    return areas


def generate_sphere_points(n_pts):
    """
    Returns list of n 3d coordinates of points on a sphere using the
    Golden Section Spiral algorithm.
    
    Parameters
    ----------
    n_pts : int
        Number of points to generate on the sphere

    Returns
    -------
    points : np.ndarray, shape=[n_points, 3]
        A set of 3D points of on a sphere
    """
    
    inc = np.pi * (3 - np.sqrt(5))
    offset = 2 / n_pts

    k = np.arange(n_pts).reshape(n_pts, 1)
    y = (k * offset - 1 + (offset / 2)).reshape(n_pts, 1)
    r = (np.sqrt(1 - y*y)).reshape(n_pts, 1)
    phi = k * inc
    points = np.concatenate((np.cos(phi)*r, y, np.sin(phi)*r), axis=1)

    return points

def annoate_traj_with_radii(traj):
    """Add the key 'Radius' to traj, with the van Der Waals radii of the atoms

    Acts in place

    Parameters
    ----------
    traj : msmbuilder.Trajectory
        A trajectory to work on
    """

    traj['Radius'] = np.zeros(len(traj['AtomNames']))
    for i, name in enumerate(traj['AtomNames']):
        name = name.strip('0123456789 ').upper()
        if name in ATOMIC_RADII:
            traj['Radius'][i] = ATOMIC_RADII[name]
        else:
            traj['Radius'][i] = ATOMIC_RADII[name[0]]
    
    
