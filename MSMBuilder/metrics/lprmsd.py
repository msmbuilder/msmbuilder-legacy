from __future__ import print_function, absolute_import, division
from mdtraj.utils.six.moves import xrange
import warnings
import numpy as np
from collections import namedtuple
import mdtraj as md

from .baseclasses import AbstractDistanceMetric


class LPRMSD(AbstractDistanceMetric):

    """
    Compute distance between frames using the Room Mean Square Deviation
    over a specifiable set of atoms using the Theobald QCP algorithm

    References
    ----------
    .. [1] Theobald, D. L. Acta. Crystallogr., Sect. A 2005, 61, 478-480.

    """
    def __init__(self, atom_indices=None, alt_indices=None, 
                 permute_groups=None, omp_parallel=True):
        """Initalize an RMSD calculator

        Parameters
        ----------
        atom_indices : array_like, optional
            List of the indices of the atoms that you want to use for the RMSD
            calculation. For example, if your trajectory contains the coordinates
            of all the atoms, but you only want to compute the RMSD on the C-alpha
            atoms, then you can supply a reduced set of atom_indices. If unsupplied,
            all of the atoms will be used.
        alt_indices : array_like, optional
            List of indices to compute the actual distance. Here, the alignment
            will be done using atom_indices, but the distance will be calculated
            between the alt_indices
        permute_groups : array_like, optional
            list of lists, where each element is a list of atom indices that are
            indistinguishable and can be permuted. The distance returned will 
            be based on finding the permutation that minimizes the total distance.
        omp_parallel : bool, optional
            Use OpenMP parallelized C code under the hood to take advantage of
            multicore architectures. If you're using another parallelization scheme
            (e.g. MPI), you might consider turning off this flag.

        Notes
        -----
        You can also control the degree of parallelism with the OMP_NUM_THREADS
        envirnoment variable


        """
        self.atom_indices = atom_indices
        self.alt_indices = alt_indices
        self.permute_groups = permute_groups

        self.omp_parallel = omp_parallel

    def __repr__(self):
        try:
            val = 'metrics.LPRMSD(atom_indices=%s, alt_indices=%s, ' % (
                    repr(list(self.atom_indices)), repr(list(self.alt_indices))) + \
                  'permute_groups=%s, omp_parallel=%s)' % (
                    repr(list(self.permute_groups)), self.omp_parallel)
        except:
            val = 'metrics.LPRMSD(atom_indices=%s, alt_indices=%s, ' % (
                    self.atom_indices, self.alt_indices) + \
                  'permute_groups=%s, omp_parallel=%s)' % (
                    self.permute_groups, self.omp_parallel)

        return val

    def prepare_trajectory(self, trajectory):
        """Prepare the trajectory for RMSD calculation.

        Preprocessing includes extracting the relevant atoms, centering the
        frames, and computing the G matrix.


        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            Molecular dynamics trajectory

        Returns
        -------
        theodata : array_like
            A msmbuilder.metrics.TheoData object, which contains some preprocessed
            calculations for the RMSD calculation
        """

        t = trajectory
        t.center_coordinates()

        return t


    def one_to_many(self, prepared_traj1, prepared_traj2, index1, indices2):
        """Calculate a vector of distances from one frame of the first trajectory
        to many frames of the second trajectory

        The distances calculated are from the `index1`th frame of `prepared_traj1`
        to the frames in `prepared_traj2` with indices `indices2`

        Parameters
        ----------
        prepared_traj1 : rmsd.TheoData
            First prepared trajectory
        prepared_traj2 : rmsd.TheoData
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory`
        indices2 : ndarray
            list of indices in `prepared_traj2` to calculate the distances to

        Returns
        -------
        Vector of distances of length len(indices2)

        Notes
        -----
        If the omp_parallel optional argument is True, we use shared-memory
        parallelization in C to do this faster. Using omp_parallel = False is
        advised if indices2 is a short list and you are paralellizing your
        algorithm (say via mpi) at a different
        level.
        """
        return self.one_to_all(prepared_traj1, prepared_traj2[indices2], index1)


    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        """Calculate a vector of distances from one frame of the first trajectory
        to all of the frames in the second trajectory

        The distances calculated are from the `index1`th frame of `prepared_traj1`
        to the frames in `prepared_traj2`

        Parameters
        ----------
        prepared_traj1 : rmsd.TheoData
            First prepared trajectory
        prepared_traj2 : rmsd.TheoData
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory`

        Returns
        -------
        Vector of distances of length len(prepared_traj2)

        Notes
        -----
        If the omp_parallel optional argument is True, we use shared-memory
        parallelization in C to do this faster.
        """

        if not self.alt_indices is None:
            distances = md.lprmsd(prepared_traj2, prepared_traj1, index1, 
                            parallel=self.omp_parallel, atom_indices=self.atom_indices, 
                            permute_groups=self.permute_groups, superpose=True)

            alt_distances = prepared_traj2.xyz[:, self.alt_indices] - prepared_traj1.xyz[index1, self.alt_indices]
            alt_distances = np.sqrt(np.square(alt_distances).sum(axis=2).mean(axis=1))
    
            return alt_distances

        else:
            distances = md.lprmsd(prepared_traj2, prepared_traj1, index1,
                            parallel=self.omp_parallel, atom_indices=self.atom_indices,
                            permute_groups=self.permute_groups)
        
            return distances 


    def _square_all_pairwise(self, prepared_traj):
        """Reference implementation of all_pairwise"""
        warnings.warn(
            'This is HORRIBLY inefficient. This operation really needs to be done directly in C')
        output = np.empty((prepared_traj.n_frames, prepared_traj.n_frames))
        for i in xrange(prepared_traj.n_frames):
            output[i] = self.one_to_all(prepared_traj, prepared_traj, i)
        return output
