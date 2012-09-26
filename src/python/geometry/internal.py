"Methods to get internal coordinates"
import numpy as np
from msmbuilder.geometry import contact
from msmbuilder.geometry import dihedral
#from msmbuilder.geometry import angles

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
    pass

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
    
    http://folk.uio.no/helgaker/reprints/2002/JCP117b_GeoOpt.pdf
    """
    pass
    
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
    pass
    
def get_dihedral_connectivity(bond_connectivity, conf):
    """Given the connectivity, get all of the bond angles
    
    Parameters
    ----------
    bond_connectivity : np.ndarray, shape=[n_bonds, 2], dtype=int
        n_bonds x 2 array of indices, where each row is the index of two
        atom who participate in a bond.
    conf : msmbuilder.Trajectory
        An msmbuilder trajectory, only the first frame will be used. This
        is used purely to make the check for angle(ABC) != 180.
    
    Returns
    -------
    dihedral_indices : np.ndarray, shape[n_dihedrals, 4], dtype=int
        All sets of 4 atoms A,B,C,D such that A is bonded to B, B is bonded
        to C, and C is bonded to D, provided that angle(ABC) != 180 and
        angle(BCD) != 180
    """
    pass
    
def get_wilson_B(conf):
    """Calculate the Wilson B matrix, which collects the derivatives of the
    redundant internal coordinates w/r/t the cartesian coordinates.

    .. math:: 

        B_{ij} = \frac{\partial q_i}{\partial x_j}

    where :math:`q_i` are the internal coorindates and the :math:`x_j` are
    the Cartesian displacement coordinates of the atoms.
    """
    pass
    