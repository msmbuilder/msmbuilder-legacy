import numpy as np
from msmbuilder import msm_analysis
from numpy import dot
inv = np.linalg.inv
norm = np.linalg.norm
tr = np.trace

import logging
logger = logging.getLogger(__name__)


class Lumper():
    def __init__(self,T,num_macrostates,flux_cutoff=None):
        """Base class for PCCA and PCCA+.
               
        Parameters
        ----------
        T : csr sparse matrix
            Transition matrix
        num_macrostates : int
            Desired number of macrostates
        flux_cutoff : float
            Can be set to discard low-flux eigenvectors.
        """
        
        
        self.T = T
        self.num_macrostates = num_macrostates        
        
        self.eigenvalues, self.left_eigenvectors = msm_analysis.get_eigenvectors(T, self.num_macrostates)
        normalize_left_eigenvectors(self.left_eigenvectors)
    
        if flux_cutoff != None:
            self.eigenvalues, self.left_eigenvectors = trim_eigenvectors_by_flux(self.eigenvalues, self.left_eigenvectors, flux_cutoff)
            self.num_macrostates = len(self.eigenvalues)
        
        self.populations = self.left_eigenvectors[:, 0]
        self.num_microstates = len(self.populations)
        
        self.right_eigenvectors = construct_right_eigenvectors(self.left_eigenvectors, self.populations, self.num_macrostates)  # Construct properly normalized right eigenvectors    
    

def construct_right_eigenvectors(left_eigenvectors, populations, num_macrostates):
    """Calculate normalized right eigenvectors from left eigenvectors and populations."""
    right_eigenvectors = left_eigenvectors.copy()
    for i in range(num_macrostates):
        right_eigenvectors[:, i] /= populations
        right_eigenvectors[:, i] *= np.sign(right_eigenvectors[0, i])
        right_eigenvectors[:, i] /= np.sqrt(dot(right_eigenvectors[:, i] * populations, right_eigenvectors[:, i]))
        
    return right_eigenvectors
    
def normalize_left_eigenvectors(left_eigenvectors):
    """Normalize the left eigenvectors
    
    Normalization condition is <left_eigenvectors[:,i] / populations, left_eigenvectors[:,i]> = 1
    
    Parameters
    ----------
    left_eigenvectors : ndarray
        The left eigenvectors, as a two-dimensional array where the kth
        eigenvectors is left_eigenvectors[:,k]
    
    Notes
    -----
    Acts inplace. Assumes that left_eigenvectors[:,0] is the equilibrium vector and that detailed balance holds.
    """
    populations = left_eigenvectors[:, 0]
    populations /= populations.sum()
    
    for k in xrange(1, left_eigenvectors.shape[-1]):
        x = left_eigenvectors[:, k]
        x /= abs(np.dot(x / populations, x)) ** .5


def trim_eigenvectors_by_flux(eigenvalues, left_eigenvectors, flux_cutoff):
    """Trim eigenvectors that have low equilibrium flux.

    Parameters
    ----------
    lam : nadarray
        Eigenvalues of transition matrix.
    vl : ndarray
        Left eigenvectors of transition matrix.
    flux_cutoff : float
        Discard eigenvectors with fluxes below this value.
    
    Notes
    -----
    Assuming that the left eigenvectors are properly pi-normalized,
    the equilibrium flux contribution of each eigenvector :math:`v` is given by :math:`\sum_i v_i^2`
    
    Returns
    -------
    lam : ndarray
        Eigenvalues after discarding low-flux eigenvectors.
    vl : ndarray
        Left eigenvectors after discarding low-flux eigenvectors.
    
    """
    normalize_left_eigenvectors(left_eigenvectors)

    N = len(eigenvalues)

    flux_list = np.array([(left_eigenvectors[:, i] ** 2).sum() for i in range(N)])
    flux_list /= flux_list[0]
    flux_list[0] = flux_list.max()
    
    KeepInd = np.where(flux_list >= flux_cutoff)[0]

    logger.info("Implied timescales (UNITLESS)")
    logger.info(-1 / np.log(eigenvalues))
    logger.info("Flux")
    logger.info(flux_list)
    logger.info("Keeping %d eigenvectors after flux cutoff %f", len(KeepInd), flux_cutoff)

    eigenvalues = eigenvalues[KeepInd]
    left_eigenvectors = left_eigenvectors[:, KeepInd]
    flux_list = flux_list[KeepInd]
    
    logger.info("After Flux calculation, Implied timescales (UNITLESS):")
    logger.info(-1 / np.log(eigenvalues))

    logger.info("After Flux calculation, fluxes.")
    logger.info(flux_list)

    return eigenvalues, left_eigenvectors    