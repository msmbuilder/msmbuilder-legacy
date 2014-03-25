"""
Code for computing cut-based free energy profiles, and optimal reaction coordinates
within that framework.

This code employs scipy's weave, so C/OMP compatability is required. Most machines
should have this functionality by default.


To Do
-----
> Choose best search method in `optimize`
> Add functionality for saving/loading the state of VariableCoordinate
"""
from __future__ import print_function, division, absolute_import
import itertools
import time

import numpy as np
import scipy
import scipy.optimize
import matplotlib.pyplot as plt

import mdtraj as md
from msmbuilder import MSMLib
from msmbuilder import tpt
from msmbuilder.msm_analysis import get_eigenvectors


TIME = False


def contact_reaction_coordinate(trajectory, weights):
    """
    Computes a residue-contact based reaction coordinate.

    Specifically, the reaction coordinate is a weighted linear combination
    of the alpha carbon distances for each residue. Weights can take any
    value, including negative and zero values.

    Parameters
    ----------
    trajectory : msmbuilder trajectory
        The trajectory along which to compute the reaction coordinate value

    weights : nd_array, float
        The weights to apply to all the inter-residue (Ca-Ca) distances

    Returns
    -------
    rc_value : nd_array, float
        The reaction coordinate value for each snapshot of `trajectory`
    """

    if TIME:
        starttime = time.clock()

    # make an array of all pairwise C-alpha indices
    C_alpha_pairs = np.array(
        list(itertools.combinations([a.index for a in t.topology.atoms if a.name == 'CA'])))

    # calculate the distance between all of those pairs
    distance_array = md.compute_distances(trajectory, C_alpha_pairs)
    rc_value = np.sum(distance_array * weights.T, axis=1)

    if TIME:
        endtime = time.clock()
        print("Time spent in RC eval", endtime - starttime)

    return rc_value


class CutCoordinate(object):

    """
    Object containing methods for computing the cut-based free energy profiles
    of reaction coordinates.

    Parameters
    ----------
    counts : matrixs
        A matrix of the transition counts observed in the MSM
    generators : msmbuilder trajectory
        The generators (or a trajectory containing an exemplar structure)
        for each state.
    reactant : int
        Index of the state to use to represent the reactant (unfolded) well
    product : int
        Index of the state to use to represent the product (folded) well
    """

    def __init__(self, counts, generators, reactant, product):
        # store the basic values
        self.counts = counts
        self.generators = generators
        self.reactant = reactant
        self.product = product
        self.N = counts.shape[0]
        self.reaction_coordinate_values = None

    def _check_coordinate_values(self):
        """
        Function that checks if we have self.reaction_coordinate_values on hand,
        and complains if we don't.
        """
        if self.reaction_coordinate_values == None:
            raise Exception("Error: No `reaction_coordinate_values` found. "
                            "Either pass in these manually or calculate them "
                            "with a class method. See the methods:"
                            "\n-- set_coordinate_values"
                            "\n-- set_coordinate_as_committors"
                            "\n-- set_coordinate_as_eigvector2")
        return

    def set_coordinate_values(self, coordinate_values):
        """
        Set the reaction coordinate manually, by providing coordinate values
        that come from some external calculation.

        Parameters
        ----------
        coordinate_values : nd_array, float
            The values of the reaction coordinate, evaluated for each state.
        """
        self.reaction_coordinate_values = coordinate_values
        self.evaluate_partition_functions()
        return

    def set_coordinate_as_committors(self, lag_time=1, symmetrize='transpose'):
        """
        Set the reaction coordinate to be the committors (pfolds).

        Employs the reactant, product states provided as the sources, sinks
        respectively for the committor calculation.

        Parameters
        ----------
        lag_time : int
            The MSM lag time to use (in units of frames) in the estimation
            of the MSM transition probability matrix from the `counts` matrix.

        symmetrize : str {'mle', 'transpose', 'none'}
            Which symmetrization method to employ in the estimation of the
            MSM transition probability matrix from the `counts` matrix.
        """

        t_matrix = MSMLib.build_msm(self.counts, symmetrize)
        self.reaction_coordinate_values = tpt.calculate_committors([self.reactant],
                                                                   [self.product],
                                                                   t_matrix)
        return

    def set_coordinate_as_eigvector2(self, lag_time=1, symmetrize='transpose'):
        """
        Set the reaction coordinate to be the second eigenvector of the MSM generated
        by counts, the provided lag_time, and the provided symmetrization method.

        Parameters
        ----------
        lag_time : int
            The MSM lag time to use (in units of frames) in the estimation
            of the MSM transition probability matrix from the `counts` matrix.

        symmetrize : str {'mle', 'transpose', 'none'}
            Which symmetrization method to employ in the estimation of the
            MSM transition probability matrix from the `counts` matrix.
        """

        t_matrix = MSMLib.build_msm_from_counts(self.counts, lag_time, symmetrize)
        v, w = get_eigenvectors(t_matrix, 5)
        self.reaction_coordinate_values = w[:, 1].flatten()

        return

    def reaction_mfpt(self, lag_time=1.0):
        """
        Calculate the MFPT between the `reactant` and `product` states, as given by
        the reaction coordinate.

        Parameters
        ----------
        lag_time : float
            The units of time for the MSM lag time.

        Returns
        -------
        mfpt : float
            The mean first passage time between the `reactant` and `product`
        """

        if TIME:
            starttime = time.clock()

        self._check_coordinate_values()

        const = lag_time / np.pi

        A = self.reaction_coordinate_values[self.reactant]
        B = self.reaction_coordinate_values[self.product]

        # perform the following integral over the rxn coord numerically
        # \int_A^B dx ( zh[x] / zc^2[x] ) * \int_A^x dy zh[y]

        perm = np.argsort(self.reaction_coordinate_values)
        start = np.where(self.reaction_coordinate_values[perm] == A)[0]
        end = np.where(self.reaction_coordinate_values[perm] == B)[0]
        intermediate_states = np.arange(self.N)[perm][start:end]

        # use weave & OMP to do the integral
        tau_parallel = 0.0
        N = len(intermediate_states)
        zc = self.zc
        zh = self.zh
        rc = self.reaction_coordinate_values

        import scipy.weave
        scipy.weave.inline(r"""
        // Loop over all `intermediate_states` to perform the integral
        // Employ a trapezoid approximation to evaluate the integral
        Py_BEGIN_ALLOW_THREADS
        int i, x, y;
        double secondint, incr, h;

        #pragma omp parallel for private(x, y, secondint, incr, h) shared(zh, zc, rc)
        // loop to perform the outer integral
        for (i = 0; i < N; i++) {

            x = intermediate_states[i];

            // loop to perform the inner/second integral (over dy)
            secondint = 0;
            for (y = 0; y < x-1; y++) {
                h = rc[y+1] - rc[y];
                secondint += h * (1.0/2.0) * (zh[y] + zh[y+1]);
            }

            // calculate the integral (over dx)
            h = rc[x+1] - rc[x];
            incr = (1.0/2.0) * h * ( ( zh[x+1] / (zc[x+1] * zc[x+1]) ) - ( zh[x] / (zc[x] * zc[x]) ) ) * secondint;

            #pragma omp critical(tau_update)
            tau_parallel += incr;
        }
        Py_END_ALLOW_THREADS
        """, ['tau_parallel', 'N', 'intermediate_states', 'zc', 'zh', 'rc'],
            extra_link_args=['-lgomp'], extra_compile_args=["-O3", "-fopenmp"])

        mfpt = const * tau_parallel

        if TIME:
            endtime = time.clock()
            print("Time spent in MFPT:", endtime - starttime)

        return mfpt

    def evaluate_partition_functions(self):
        """
        Computes the partition function of the cut-based free energy profile based
        on transition network and reaction coordinate.

        Employs an MSM to do so, thus taking input in the form of a discrete state
        space and the observed dynamics in that space (assignments). Further, one
        can provide a "reaction coordiante" who's values will be employed to
        find the min-cut/max-flow coordinate describing dynamics on that space.

        Generously contributed by Sergei Krivov
        Optimizations and modifications due to TJL

        Stores
        ------
        zc : nd_array, float
            The cut-based free energy profile along the reaction coordinate. The
            negative log of this function gives the free energy profile along the
            coordinate.
        zh : nd_array, float
            The histogram-based free energy profile.

        See Also
        --------
        optimize_cut_based_coordinate : function
            Optimize a flexible reaction coordinate to maximize the free energy
            barrier along that coordinate.
        """

        if TIME:
            starttime = time.clock()
        self._check_coordinate_values()

        # permute the counts matrix to order it with the rxn coordinate
        state_order_along_RC = np.argsort(self.reaction_coordinate_values)
        perm_counts = MSMLib.permute_mat(self.counts, state_order_along_RC)

        # set up the three variables for weave -- need to be locals
        data = perm_counts.data
        indices = perm_counts.indices
        indptr = perm_counts.indptr

        zc = np.zeros(self.N)
        N = self.N

        import scipy.weave
        scipy.weave.inline(r"""
        Py_BEGIN_ALLOW_THREADS
        int i, j, k;
        double nij, incr;

        #pragma omp parallel for private(nij, incr, j, k) shared(N, indptr, indices, data, zc)
        for (i = 0; i < N; i++) {
            for (k = indptr[i]; k < indptr[i+1]; k++) {
                j = indices[k];
                if (i == j) { continue; }

                nij = data[k];

                // iterate through all values in the matrix in csr format
                // i, j are the row and column indices
                // nij is the entry

                incr = j < i ? nij : -nij;

                #pragma omp critical(zc_update)
                {
                    zc[j] += incr;
                    zc[i] -= incr;
                }
            }
        }
        Py_END_ALLOW_THREADS
        """, ['N', 'data', 'indices', 'indptr', 'zc'], extra_link_args=['-lgomp'],
            extra_compile_args=["-O3", "-fopenmp"])

        zc /= 2.0  # we overcounted in the above - fix that

        # put stuff back in the original order
        inv_ordering = np.argsort(state_order_along_RC)
        zc = zc[inv_ordering]

        # calculate the histogram-based partition function
        zh = np.array(self.counts.sum(axis=0)).flatten()

        self.zc = zc
        self.zh = zh

        if TIME:
            endtime = time.clock()
            print("Time spent in zc:", endtime - starttime)

        return

    def rescale_to_natural_coordinate(self):
        """
        Rescale a cut-based free energy profile along a reaction coordinate
        such that the diffusion constant is unity for the entire coordinate.

        Parameters
        ----------
        zc : ndarray, float
            The cut-based partition function from `calc_cFEP`
        zh : ndarray, float
            The histogram-based parition function from `calc_cFEP`
        rxn_coordinate : ndarray, float
            The reaction coordinate used.

        Updates
        -------
        zc, zh <- natural_coordinate : ndarray, float
            The reaction coordinate values along the rescaled coordinate
        reaction_coordinate_values <- scaled_z : ndarray, float
            The partition function value for each point of `natural_coordinate`
        """

        self._check_coordinate_values()

        state_order_along_RC = np.argsort(self.reaction_coordinate_values)
        zc = self.zc[state_order_along_RC]
        zh = self.zh[state_order_along_RC]

        scaled_z = np.cumsum(zc)
        positive_inds = np.where(scaled_z > 0.0)
        scaled_z = scaled_z[positive_inds]
        natural_coordinate = np.cumsum(zh[positive_inds] / (scaled_z * np.sqrt(np.pi)))

        self.zc = natural_coordinate
        self.zh = natural_coordinate
        self.reaction_coordinate_values = scaled_z

        return

    def plot(self, num_bins=15, filename=None):
        """
        Plot the current cut-based free energy profile.

        Can either display this on screen (filename=None), or save to a file
        if the kwarg `filename` is provided.

        The plot does a smoothing average over the reaction coordinate.
        Increasing `bins` makes the average finer, and the resulting coordinate
        rougher. Fewer `bins` means a coarser, smoother coordinate.

        Parameters
        ----------
        num_bins : int
            The number of bins to employ in a sliding average.
        filename : str
            The name of the file to save the rendered image to. If None is
            passed, attempts to display the image on screen.
        """

        self._check_coordinate_values()

        perm = np.argsort(self.reaction_coordinate_values)

        # performing a sliding average
        h = self.reaction_coordinate_values.max() / float(num_bins)
        bins = [i * h for i in range(num_bins)]

        inds = np.digitize(self.reaction_coordinate_values, bins)
        pops = [np.sum(self.zc[np.where(inds == i)]) for i in range(num_bins)]

        fig = plt.figure()
        plt.plot(bins, -1.0 * np.log(pops), lw=2)
        plt.xlabel('Reaction Coordinate')
        plt.ylabel('Free Energy Profile (kT)')

        if not filename:
            plt.show()
        else:
            plt.savefig(filename)
            print("Saved reaction coordinate plot to: %s" % filename)

        return


class VariableCoordinate(CutCoordinate):

    """
    Class that contains methods for calculating cut-based free energy profiles
    based on a reaction coordinate mapping that takes variable parameters.

    Specifically, a reaction coodinate is formally a map from phase space to
    the reals. Consider such a map that is dependent on some auxillary parameters
    `alphas`. Then we might hope to optimize that reaction coordinate by wisely
    choosing those weights.

    Parameters
    ----------
    rxn_coordinate_function : function
        This is a function that represents the reaction coordinate. It
        takes the arguments of an msm trajectory and an array of weights
        (floats). It should return an array of floats, specifically the
        reaction coordinate scalar evaluated for each conformation in the
        trajectory. Graphically:

        rxn_coordinate_function(msm_traj, weights) --> array( [float, float, ...] )

        An example function of this form is provided, for contact-map based
        reaction coordinates.
    initial_alphas : nd_array, float
        An array representing an initial guess at the optimal alphas.
    counts : matrixs
        A matrix of the transition counts observed in the MSM
    generators : msmbuilder trajectory
        The generators (or a trajectory containing an exemplar structure)
        for each state.
    reactant : int
        Index of the state to use to represent the reactant (unfolded) well
    product : int
        Index of the state to use to represent the product (folded) well
    """

    def __init__(self, rxn_coordinate_function, initial_alphas, counts,
                 generators, reactant, product):
        CutCoordinate.__init__(self, counts, generators, reactant, product)
        self.rxn_coordinate_function = rxn_coordinate_function
        self.rc_alphas = initial_alphas
        self._evaluate_reaction_coordinate()

    def _evaluate_reaction_coordinate(self):
        """
        Evaluates `rxn_coordinate_function` and stores the values obtained in a
        local variable.
        """
        self.reaction_coordinate_values = self.rxn_coordinate_function(
            self.generators, self.rc_alphas)
        return

    def optimize(self, maxiter=1000):
        """
        Compute an optimized reaction coordinate, where optimized means the
        coordinate the maximizes the barrier between two metastable basins.

        Parameters
        ----------
        maxiter : int (optional)
            The maximum number of iterations to run the optimization algorithm
            for. Note that you can always do a little, check the answer, then 
            come back and optimize more later.

        Stores
        ------
        optimal_weights : nd_array, float
            An array of the optimal weights, which maximize the barrier in the
            cut-based free energy profile.
        """

        self.i = 1

        def objective(weights, generators):
            """ returns the negative of the MFPT """

            starttime = time.clock()

            self._evaluate_reaction_coordinate()
            self.evaluate_partition_functions()
            mfpt = self.reaction_mfpt(lag_time=1.0)

            endtime = time.clock()
            if TIME:
                "Iteration %d, time: %f" % (self.i, endtime - starttime)
            self.i += 1

            return -1.0 * mfpt

        optimal_alphas = scipy.optimize.fmin_cg(objective, self.rc_alphas,
                                                args=(self.generators,),
                                                maxiter=maxiter)

        self.rc_alphas = optimal_alphas
        self._evaluate_reaction_coordinate()

        return


def test():
    from scipy import io
    import mdtraj

    print("Testing cfep code....")

    test_dir = '/Users/TJ/Programs/msmbuilder.sandbox/tjlane/cfep/'

    generators = mdtraj.load(test_dir + 'Gens.h5')
    counts = io.mmread(test_dir + 'tCounts.mtx')
    reactant = 0    # generator w/max RMSD
    product = 10598  # generator w/min RMSD
    pfolds = np.loadtxt(test_dir + 'FCommittors.dat')

    # test the usual coordinate
    #pfold_cfep = CutCoordinate(counts, generators, reactant, product)
    # pfold_cfep.set_coordinate_values(pfolds)
    # pfold_cfep.plot()

    # pfold_cfep.set_coordinate_as_eigvector2()
    # print pfold_cfep.reaction_coordinate_values
    # pfold_cfep.plot()

    # pfold_cfep.set_coordinate_as_committors()
    # print pfold_cfep.reaction_coordinate_values
    # pfold_cfep.plot()

    # test the Variable Coordinate
    initial_weights = np.ones((1225, 26104))

    contact_cfep = VariableCoordinate(contact_reaction_coordinate, initial_weights,
                                      counts, generators, reactant, product)

    contact_cfep.evaluate_partition_functions()
    print(contact_cfep.zh)
    print(contact_cfep.zc)

    contact_cfep.optimize()
    print("Finished optimization")

    contact_cfep.plot()

    return

if __name__ == '__main__':
    test()
