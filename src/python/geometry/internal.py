"Methods to get internal coordinates"
import numpy as np
import IPython as ip
from itertools import combinations, ifilter

from msmbuilder.geometry import contact
from msmbuilder.geometry import dihedral
from msmbuilder.geometry import angle
from scipy.spatial.distance import squareform, pdist
import networkx as nx


# these are covalent radii taken from the crystalographic data in nm
# Dalton Trans., 2008, 2832-2838, DOI: 10.1039/B801115J
# http://pubs.rsc.org/en/Content/ArticleLanding/2008/DT/b801115j
COVALENT_RADII = {'C': 0.0762, 'N': 0.0706, 'O': 0.0661, 'H': 0.031,
                  'S': 0.105}


def get_redundant_internal_coordinates(xyzlist, ibonds, iangles, idihedrals):
    """Compute internal coordinates from the cartesian coordinates
    
    Parameters
    ----------
    xyzlist : np.ndarray, shape=[n_frames, n_atoms, 3], dtype=float
        The cartesian coordinates
    ibonds : np.ndarray, shape[n_bonds, 2], dtype=int
        Each row gives the indices of two atoms involved in a bond
    iangles : np.ndarray, shape[n_angles, 3], dtype=int
        Each row gives the indices of three atoms which together make an angle
    idihedrals : np.ndarray, shape[n_dihedrals, 4], dtype=int
        Each row gives the indices of the four atoms which together make a
        dihedral
    
    Returns
    -------
    internal_coords : np.ndarray, shape=[n_frames, n_bonds+n_angles+n_dihedrals]
        All of the internal coordinates collected into a big array, such that
        internal_coords[i,j] gives the jth coordinate for the ith frame.
    """
    
    # convert everything to the right shape and C ordering, since
    # all of these methods are in C and are going to need things to be
    # the right type. The methods will all do a copy for things that
    # aren't the right type, but hopefully we can only do the copy once
    # instead of three times if xyzlist really does need to be reordered
    # in memory
    xyzlist = np.array(xyzlist, dtype=np.float32, order='c')
    ibonds = np.array(ibonds, dtype=np.int32, order='c')
    iangles = np.array(iangles, dtype=np.int32, order='c')
    idihedrals = np.array(idihedrals, dtype=np.int32, order='c')
    
    b = contact.atom_distances(xyzlist, ibonds)
    a = angle.bond_angles(xyzlist, iangles)
    d = dihedral.compute_dihedrals(xyzlist, idihedrals, degrees=False)
    
    return np.hstack((b,a,d))


def get_bond_connectivity(conf):
    """
    Get the connectivity of a conformation
    
    Parameters
    ----------
    conf : msmbuilder.Trajectory
        An msmbuilder trajectory, only the first frame will be used.
    
    Returns
    -------
    connectivity : np.ndarray, shape=[n_bonds, 2], dtype=int
        n_bonds x 2 array of indices, where each row is the index of two
        atom who participate in a bond.
    
    Notes
    -----
    Regular bonds are assigned to all pairs of atoms where
    the interatomic distance is less than or equal to 1.3 times the
    sum of their respective covalent radii.
    
    Bakken and Helgaker, JCP Vol. 117, Num. 20 22 Nov. 2002
    http://folk.uio.no/helgaker/reprints/2002/JCP117b_GeoOpt.pdf
    """
    
    xyz = conf['XYZList'][0, :, :]
    n_atoms = xyz.shape[0]
    
    elements = np.zeros(n_atoms, dtype='S1')
    for i in xrange(n_atoms):
        # name of the element that is atom[i]
        # take the first character of the AtomNames string,
        # after stripping off any digits
        elements[i] = conf['AtomNames'][i].strip('123456789 ')[0]
        if not elements[i] in COVALENT_RADII.keys():
            raise ValueError("I don't know about this AtomName: {}".format(
                conf['AtomNames'][i]))

    distance_mtx = squareform(pdist(xyz))
    connectivity = []

    for i in xrange(n_atoms):
        for j in xrange(i+1, n_atoms):
            # Regular bonds are assigned to all pairs of atoms where
            # the interatomic distance is less than or equal to 1.3 times the
            # sum of their respective covalent radii.
            d = distance_mtx[i, j]
            if d < 1.3 * (COVALENT_RADII[elements[i]] + COVALENT_RADII[elements[j]]):
                connectivity.append((i, j))

    return np.array(connectivity)


def get_angle_connectivity(bond_connectivity):
    """Given the connectivity, get all of the bond angles
    
    Parameters
    ----------
    bond_connectivity : np.ndarray, shape=[n_bonds, 2], dtype=int
        n_bonds x 2 array of indices, where each row is the index of two
        atom who participate in a bond.
    
    Returns
    -------
    angle_indices : np.ndarray, shape[n_angles, 3], dtype=int
        n_angles x 3 array of indices, where each row is the index of three
        atoms m,n,o such that n is bonded to both m and o.
    """

    graph = nx.from_edgelist(bond_connectivity)
    n_atoms = graph.number_of_nodes()
    angle_indices = []

    for i in xrange(n_atoms):
        for (m, n) in combinations(graph.neighbors(i), 2):
            # so now the there is a bond angle m-i-n
            angle_indices.append((m,i,n))

    return np.array(angle_indices)


def get_dihedral_connectivity(bond_connectivity,):
    """Given the connectivity, get all of the bond angles
    
    Parameters
    ----------
    bond_connectivity : np.ndarray, shape=[n_bonds, 2], dtype=int
        n_bonds x 2 array of indices, where each row is the index of two
        atom who participate in a bond.
    #conf : msmbuilder.Trajectory
    #    An msmbuilder trajectory, only the first frame will be used. This
    #    is used purely to make the check for angle(ABC) != 180.
    
    Returns
    -------
    dihedral_indices : np.ndarray, shape[n_dihedrals, 4], dtype=int
        All sets of 4 atoms A,B,C,D such that A is bonded to B, B is bonded
        to C, and C is bonded to D
    """
    graph = nx.from_edgelist(bond_connectivity)
    n_atoms = graph.number_of_nodes()
    dihedral_indices = []

    for a in xrange(n_atoms):
        for b in graph.neighbors(a):
            for c in ifilter(lambda c: c not in [a,b], graph.neighbors(b)):
                for d in ifilter(lambda d: d not in [a,b,c], graph.neighbors(c)):
                    dihedral_indices.append((a,b,c,d))

    return np.array(dihedral_indices)


def get_wilson_B(conf):
    """Calculate the Wilson B matrix, which collects the derivatives of the
    redundant internal coordinates w/r/t the cartesian coordinates.

    .. math:: 

        B_{ij} = \frac{\partial q_i}{\partial x_j}

    where :math:`q_i` are the internal coorindates and the :math:`x_j` are
    the Cartesian displacement coordinates of the atoms.
    """
    pass


if __name__ == '__main__':
    from msmbuilder import Trajectory
    conf = Trajectory.load_trajectory_file('Tutorial/native.pdb')
    b = get_bond_connectivity(conf)
    a = get_angle_connectivity(b)
    d = get_dihedral_connectivity(b)
    
    coords = get_redundant_internal_coordinates(conf['XYZList'], b, a, d)
    
    ip.embed()





    