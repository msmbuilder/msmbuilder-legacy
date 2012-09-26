"Methods to get internal coordinates"
import numpy as np
import IPython as ip
from itertools import combinations, ifilter

from msmbuilder.geometry.contact import atom_distances
from msmbuilder.geometry.dihedral import compute_dihedrals
from msmbuilder.geometry.angle import bond_angles
from scipy.spatial.distance import squareform, pdist
import networkx as nx


# these are covalent radii taken from the crystalographic data in nm
# Dalton Trans., 2008, 2832-2838, DOI: 10.1039/B801115J
# http://pubs.rsc.org/en/Content/ArticleLanding/2008/DT/b801115j
COVALENT_RADII = {'C': 0.0762, 'N': 0.0706, 'O': 0.0661, 'H': 0.031,
                  'S': 0.105}


################################################################################
# Get actual coordinates
################################################################################


def get_redundant_internal_coordinates(trajectory):
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
    
    ibonds, iangles, idihedrals = get_connectivity(trajectory)
    
    # convert everything to the right shape and C ordering, since
    # all of these methods are in C and are going to need things to be
    # the right type. The methods will all do a copy for things that
    # aren't the right type, but hopefully we can only do the copy once
    # instead of three times if xyzlist really does need to be reordered
    # in memory
    
    xyzlist = np.array(trajectory['XYZList'], dtype=np.float32, order='c')
    ibonds = np.array(ibonds, dtype=np.int32, order='c')
    iangles = np.array(iangles, dtype=np.int32, order='c')
    idihedrals = np.array(idihedrals, dtype=np.int32, order='c')
    
    b = atom_distances(xyzlist, ibonds)
    a = bond_angles(xyzlist, iangles)
    d = compute_dihedrals(xyzlist, idihedrals, degrees=False)
    
    return np.hstack((b,a,d))


def get_wilson_B(conformation):
    """Calculate the Wilson B matrix, which collects the derivatives of the
    redundant internal coordinates w/r/t the cartesian coordinates.

    .. math:: 

        B_{ij} = \frac{\partial q_i}{\partial x_j}

    where :math:`q_i` are the internal coorindates and the :math:`x_j` are
    the Cartesian displacement coordinates of the atoms.
    """
    xyz = conformation['XYZList'][0]
    
    ibonds, iangles, idihedrals = get_connectivity(conformation)
    
    bd = get_bond_derivs(xyz, ibonds)
    ad = get_angle_derivs(xyz, iangles)
    dd = get_dihedral_derivs(xyz, idihedrals)
    
    return np.vstack((bd, ad, dd))



################################################################################
# Compte the connectivity, getting lists of atom indices which form bonds, bond
# angles and dihedrals
################################################################################

def get_connectivity(conf):
    "Convenience method"
    ibonds = get_bond_connectivity(conf)
    iangles = get_angle_connectivity(ibonds)
    idihedrals = get_dihedral_connectivity(ibonds)

    return ibonds, iangles, idihedrals


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


################################################################################
# Compute derivatives of internal coordinates w.r.t to cartesian coordinates
# these methods only operate on a single frame
################################################################################

def get_bond_derivs(xyz, ibonds):
    n_atoms, n_bonds = xyz.shape[0], len(ibonds)
    
    derivatives = np.zeros((n_bonds, n_atoms, 3))
    for b, (m, n) in enumerate(ibonds):
        u = (xyz[m] - xyz[n]) / np.linalg.norm(xyz[m] - xyz[n])

        derivatives[b, m, :] = u
        derivatives[b, n, :] = -u

    return derivatives


def get_angle_derivs(xyz, iangles):
    n_atoms, n_angles = xyz.shape[0], len(iangles)

    derivatives = np.zeros((n_angles, n_atoms, 3))
    vector1 = np.array([1, -1, 1]) / np.sqrt(3)
    vector2 = np.array([-1, 1, 1]) / np.sqrt(3)

    for a, (m, o, n) in enumerate(iangles):
        u_prime = (xyz[m] - xyz[o])
        u_norm = np.linalg.norm(u_prime)
        v_prime = (xyz[n] - xyz[o])
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        v = v_prime / v_norm

        if np.linalg.norm(u + v) < 1e-10 or np.linalg.norm(u - v) < 1e-10:
            # if they're parallel            
            if np.linalg.norm(u + vector1) < 1e-10 or np.linalg.norm(u - vector1) < 1e-10:
                # and they're parallel o [1, -1, 1]
                w_prime = np.cross(u, vector2)
            else:
                w_prime = np.cross(u, vector1)
        else:
             w_prime = np.cross(u, v)
             
        w = w_prime / np.linalg.norm(w_prime)


        derivatives[a, m, :] = np.cross(u, w) / u_norm
        derivatives[a, n, :] = np.cross(w, v) / v_norm
        derivatives[a, o, :] = -np.cross(u, w) / u_norm - np.cross(w, v) / v_norm

    return derivatives


def get_dihedral_derivs(xyz, idihedrals):
    n_atoms, n_dihedrals = xyz.shape[0], len(idihedrals)

    derivatives = np.zeros((n_dihedrals, n_atoms, 3))
    vector1 = np.array([1, -1, 1]) / np.sqrt(3)
    vector2 = np.array([-1, 1, 1]) / np.sqrt(3)
    
    for d, (m, o, p, n) in enumerate(idihedrals):
        u_prime = (xyz[m] - xyz[o])
        w_prime = (xyz[p] - xyz[o])
        v_prime = (xyz[n] - xyz[p])
        
        u_norm = np.linalg.norm(u_prime)
        w_norm = np.linalg.norm(w_prime)
        v_norm = np.linalg.norm(v_prime)
        
        u = u_prime / u_norm
        w = w_prime / w_norm
        v = v_prime / v_norm
        
        term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
        term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
        term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        term4 = np.cross(v, w) * -np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))

        derivatives[d, m, :] = term1
        derivatives[d, n, :] = -term2
        derivatives[d, o, :] = -term1 + term3 - term4
        derivatives[d, p, :] = term2 - term3 + term4
    
    return derivatives



if __name__ == '__main__':
    from msmbuilder import Trajectory
    h = 1e-4
    conf1 = Trajectory.load_trajectory_file('Tutorial/native.pdb')
    conf2 = Trajectory.load_trajectory_file('Tutorial/native.pdb')
    conf2['XYZList'][0, 0, 0] += h
    
    internal1 = get_redundant_internal_coordinates(conf1)[0]
    internal2 = get_redundant_internal_coordinates(conf2)[0]

    fd = (internal2 - internal1) / h

    # n_internal_coords, n_atoms, n_dims 
    b1 = get_wilson_B(conf1)[:, 0, 0]



    ip.embed()
    print fd - b1


    