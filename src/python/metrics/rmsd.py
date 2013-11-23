import warnings
import numpy as np
from collections import namedtuple
from baseclasses import AbstractDistanceMetric
from msmbuilder import _rmsdcalc


class RMSD(AbstractDistanceMetric):
    """
    Compute distance between frames using the Room Mean Square Deviation
    over a specifiable set of atoms using the Theobald QCP algorithm

    References
    ----------
    .. [1] Theobald, D. L. Acta. Crystallogr., Sect. A 2005, 61, 478-480.

    """

    class TheoData(object):
        """Stores temporary data required during Theobald RMSD calculation.

        Notes:
        Storing temporary data allows us to avoid re-calculating the G-Values
        repeatedly. Also avoids re-centering the coordinates."""

        Theoslice = namedtuple('TheoSlice', ('xyz', 'G'))
        
        def __init__(self, XYZData, NumAtoms=None, G=None):
            """Create a container for intermediate values during RMSD Calculation.

            Notes:
            1.  We remove center of mass.
            2.  We pre-calculate matrix magnitudes (ConfG)"""

            if NumAtoms is None or G is None:
                NumConfs = len(XYZData)
                NumAtoms = XYZData.shape[1]

                self.centerConformations(XYZData)

                NumAtomsWithPadding = 4 + NumAtoms - NumAtoms % 4

                # Load data and generators into aligned arrays
                XYZData2 = np.zeros((NumConfs, 3, NumAtomsWithPadding), dtype=np.float32)
                for i in range(NumConfs):
                    XYZData2[i, 0:3, 0:NumAtoms] = XYZData[i].transpose()

                #Precalculate matrix magnitudes
                ConfG = np.zeros((NumConfs,),dtype=np.float32)
                for i in xrange(NumConfs):
                    ConfG[i] = self.calcGvalue(XYZData[i, :, :])

                self.XYZData = XYZData2
                self.G = ConfG
                self.NumAtoms = NumAtoms
                self.NumAtomsWithPadding = NumAtomsWithPadding
                self.CheckCentered()
            else:
                self.XYZData = XYZData
                self.G = G
                self.NumAtoms = NumAtoms
                self.NumAtomsWithPadding = XYZData.shape[2]

        def __getitem__(self, key):
            # to keep the dimensions right, we make everything a slice
            if isinstance(key, int):
                key = slice(key, key+1)
            return RMSD.TheoData(self.XYZData[key], NumAtoms=self.NumAtoms, G=self.G[key])

        def __setitem__(self, key, value):
            self.XYZData[key] = value.XYZData
            self.G[key] = value.G

        def CheckCentered(self, Epsilon=1E-5):
            """Raise an exception if XYZAtomMajor has nonnzero center of mass(CM)."""

            XYZ = self.XYZData.transpose(0, 2, 1)
            x = np.array([max(abs(XYZ[i].mean(0))) for i in xrange(len(XYZ))]).max()
            if x > Epsilon:
                raise Exception("The coordinate data does not appear to have been centered correctly.")

        @staticmethod
        def centerConformations(XYZList):
            """Remove the center of mass from conformations.  Inplace to minimize mem. use."""

            for ci in xrange(XYZList.shape[0]):
                X = XYZList[ci].astype('float64')  # To improve the accuracy of RMSD, it can help to do certain calculations in double precision.
                X -= X.mean(0)
                XYZList[ci] = X.astype('float32')
            return

        @staticmethod
        def calcGvalue(XYZ):
            """Calculate the sum of squares of the key matrix G.  A necessary component of Theobold RMSD algorithm."""

            conf=XYZ.astype('float64')  # Doing this operation in double significantly improves numerical precision of RMSD
            G = 0
            G += np.dot(conf[:, 0], conf[:, 0])
            G += np.dot(conf[:, 1], conf[:, 1])
            G += np.dot(conf[:, 2], conf[:, 2])
            return G

        def __len__(self):
            return len(self.XYZData)

    def __init__(self, atomindices=None, omp_parallel=True):
        """Initalize an RMSD calculator

        Parameters
        ----------
        atomindices : array_like, optional
            List of the indices of the atoms that you want to use for the RMSD
            calculation. For example, if your trajectory contains the coordinates
            of all the atoms, but you only want to compute the RMSD on the C-alpha
            atoms, then you can supply a reduced set of atom_indices. If unsupplied,
            all of the atoms will be used.
        omp_parallel : bool, optional
            Use OpenMP parallelized C code under the hood to take advantage of
            multicore architectures. If you're using another parallelization scheme
            (e.g. MPI), you might consider turning off this flag.

        Notes
        -----
        You can also control the degree of parallelism with the OMP_NUM_THREADS
        envirnoment variable


        """
        self.atomindices = atomindices
        self.omp_parallel = omp_parallel

    def __repr__(self):
        try:
            val = 'metrics.RMSD(atom_indices=%s, omp_parallel=%s)' % (repr(list(self.atomindices)), self.omp_parallel)
        except:
            val = 'metrics.RMSD(atom_indices=%s, omp_parallel=%s)' % (self.atomindices, self.omp_parallel)
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

        if self.atomindices is not None:
            return self.TheoData(trajectory.xyz[:,self.atomindices])
        return self.TheoData(trajectory.xyz)

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

        if isinstance(indices2, list):
            indices2 = np.array(indices2)
        if not isinstance(prepared_traj1, RMSD.TheoData):
            raise TypeError('Theodata required')
        if not isinstance(prepared_traj2, RMSD.TheoData):
            raise TypeError('Theodata required')

        if self.omp_parallel:
            return _rmsdcalc.getMultipleRMSDs_aligned_T_g_at_indices(
                      prepared_traj1.NumAtoms, prepared_traj1.NumAtomsWithPadding,
                      prepared_traj1.NumAtomsWithPadding, prepared_traj2.XYZData,
                      prepared_traj1.XYZData[index1], prepared_traj2.G,
                      prepared_traj1.G[index1], indices2)
        else:
            return _rmsdcalc.getMultipleRMSDs_aligned_T_g_at_indices_serial(
                      prepared_traj1.NumAtoms, prepared_traj1.NumAtomsWithPadding,
                      prepared_traj1.NumAtomsWithPadding, prepared_traj2.XYZData,
                      prepared_traj1.XYZData[index1], prepared_traj2.G,
                      prepared_traj1.G[index1], indices2)

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

        if self.omp_parallel:
            return _rmsdcalc.getMultipleRMSDs_aligned_T_g(
                prepared_traj1.NumAtoms, prepared_traj1.NumAtomsWithPadding,
                prepared_traj1.NumAtomsWithPadding, prepared_traj2.XYZData,
                prepared_traj1.XYZData[index1], prepared_traj2.G,
                prepared_traj1.G[index1])
        else:
            return _rmsdcalc.getMultipleRMSDs_aligned_T_g_serial(
                    prepared_traj1.NumAtoms, prepared_traj1.NumAtomsWithPadding,
                    prepared_traj1.NumAtomsWithPadding, prepared_traj2.XYZData,
                    prepared_traj1.XYZData[index1], prepared_traj2.G,
                    prepared_traj1.G[index1])

    def _square_all_pairwise(self, prepared_traj):
        """Reference implementation of all_pairwise"""
        warnings.warn('This is HORRIBLY inefficient. This operation really needs to be done directly in C')
        traj_length = prepared_traj.XYZData.shape[0]
        output = np.empty((traj_length, traj_length))
        for i in xrange(traj_length):
            output[i] = self.one_to_all(prepared_traj, prepared_traj, i)
        return output
