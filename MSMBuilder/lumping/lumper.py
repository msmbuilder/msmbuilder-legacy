from __future__ import print_function, absolute_import, division

import numpy as np
from msmbuilder import msm_analysis
from msmbuilder.lumping import utils
import logging
logger = logging.getLogger(__name__)


class Lumper():  # Not sure what the most general lumper would look like, so fill in later.
    pass


class EigenvectorLumper(Lumper):

    def __init__(self, T, num_macrostates, flux_cutoff=None):
        """Base class for PCCA and PCCA+.

        Parameters
        ----------
        T : csr sparse matrix
            Transition matrix
        num_macrostates : int
            Desired number of macrostates
        flux_cutoff : float, optional
            Can be set to discard low-flux eigenvectors.
        """

        self.T = T
        self.num_macrostates = num_macrostates

        self.eigenvalues, self.left_eigenvectors = msm_analysis.get_eigenvectors(T, self.num_macrostates)
        utils.normalize_left_eigenvectors(self.left_eigenvectors)

        if flux_cutoff != None:
            self.eigenvalues, self.left_eigenvectors = utils.trim_eigenvectors_by_flux(
                self.eigenvalues, self.left_eigenvectors, flux_cutoff)
            self.num_macrostates = len(self.eigenvalues)

        self.populations = self.left_eigenvectors[:, 0]
        self.num_microstates = len(self.populations)

        # Construct properly normalized right eigenvectors
        self.right_eigenvectors = utils.construct_right_eigenvectors(
            self.left_eigenvectors, self.populations, self.num_macrostates)  
