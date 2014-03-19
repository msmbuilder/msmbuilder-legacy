from __future__ import print_function, absolute_import, division

import numpy as np

from .lumper import EigenvectorLumper

import logging
logger = logging.getLogger(__name__)


class PCCA(EigenvectorLumper):

    def __init__(self, T, num_macrostates, tolerance=1E-5, flux_cutoff=None):
        """Create a lumped model using the PCCA algorithm.

        1.  Iterate over the eigenvectors, starting with the slowest.
        2.  Calculate the spread of that eigenvector within each existing macrostate.
        3.  Pick the macrostate with the largest eigenvector spread.
        4.  Split the macrostate based on the sign of the eigenvector.

        Parameters
        ----------
        T : csr sparse matrix
            A transition matrix
        num_macrostates : int
            The desired number of states.
        tolerance : float, optional
            Specifies the numerical cutoff to use when splitting states based on sign.
        flux_cutoff : float, optional
            If enabled, discard eigenvectors with flux below this value.

        Returns
        -------
        microstate_mapping : ndarray
            mapping from the Microstate indices to the Macrostate indices

        Notes
        -------
        To construct a Macrostate MSM, you then need to map your Assignment data to
        the new states (e.g. MSMLib.apply_mapping_to_assignments).

        References
        ----------
        .. [1]  Deuflhard P, et al.  "Identification of almost invariant
        aggregates in reversible nearly uncoupled markov chains,"
        Linear Algebra Appl., vol 315 pp 39-59, 2000.
        """
        EigenvectorLumper.__init__(self, T, num_macrostates, flux_cutoff=None)
        self.lump(tolerance=tolerance)

    def lump(self, tolerance):
        """Do the PCCA lumping.

        Notes
        -------
        1.  Iterate over the eigenvectors, starting with the slowest.
        2.  Calculate the spread of that eigenvector within each existing macrostate.
        3.  Pick the macrostate with the largest eigenvector spread.
        4.  Split the macrostate based on the sign of the eigenvector.
        """

        right_eigenvectors = self.right_eigenvectors[:, 1:]  # Extract non-perron eigenvectors

        microstate_mapping = np.zeros(self.num_microstates, 'int')

        # Function to calculate the spread of a single eigenvector.
        spread = lambda x: x.max() - x.min()

        for i in range(self.num_macrostates - 1):  # Thus, if we want 2 states, we split once.
            v = right_eigenvectors[:, i]
            all_spreads = np.array([spread(v[microstate_mapping == k]) for k in range(i + 1)])
            state_to_split = np.argmax(all_spreads)
            microstate_mapping[(microstate_mapping == state_to_split) & (v >= tolerance)] = i + 1

        self.microstate_mapping = microstate_mapping
