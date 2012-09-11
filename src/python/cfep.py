

import numpy as np
import scipy
import scipy.optimize
import scipy.weave
import matplotlib.pyplot as plt

from msmbuilder import MSMLib
from msmbuilder import Serializer
from msmbuilder.geometry.contact import atom_distances


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
    
    # make an array of all pairwise C-alpha indices
    C_alphas = np.where( trajectory['AtomNames'] == 'CA' )[0] # indices of the Ca atoms
    n_residues = len(C_alphas)
    atom_contacts = np.zeros(( n_residues, 2 ))
    atom_contacts[:,0] = np.repeat( C_alphas, n_residues )
    atom_contacts[:,1] = np.tile( C_alphas, n_residues )
    
    # calculate the distance between all of those pairs
    distance_array = atom_distances(trajectory['XYZList'], atom_contacts)
    rc_value = np.sum( distance_array * weights.T, axis=1 )
    
    return rc_value


class CutCoordinate(object):


    def __init__(self, rxn_coordinate_function, counts, generators, reactant, product):
        """
        Parameters
        ----------
        generators : msmbuilder trajectory
            The generators (or a trajectory containing an exemplar structure)
            for each state.

        reactant : int
            Index of the state to use to represent the reactant (unfolded) well

        product : int
            Index of the state to use to represent the product (folded) well

        coordinate_fxn : function
            This is a function that represents the reaction coordinate. It
            takes the arguments of an msm trajectory and an array of weights
            (floats). It should return an array of floats, specifically the
            reaction coordinate scalar evaluated for each conformation in the
            trajectory. Graphically:

              coordinate_fxn(msm_traj, weights) --> array( [float, float, ...] )

            An example function of this form is provided, for contact-map based
            reaction coordinates.
        """
        
        # store the basic values
        self.counts = counts
        self.generators = generators
        self.reactants = reactant
        self.products = product
        
        # perform basic calculations for stuff we'll need
        self.N = assignments.max() + 1
    
        self.rxn_coordinate_function = rxn_coordinate_function
        self.rc_weights = np.ones(self.N) # TJL: this may vary... how can we be smart about it?
        self.rxn_coordinate_values = self.rxn_coordinate_function(self.generators, self.rc_weights)


    def optimize(self, initial_weights=None):
        """
        Compute an optimized reaction coordinate, where optimized means the
        coordinate the maximizes the barrier between two metastable basins.

        Parameters
        ----------
        initial_weights : nd_array, float
            An array representing an initial guess at the optimal weights. 
            Passing `None` simply guesses uniform weights, if no prior information
            is available.

        Returns
        -------
        optimal_weights : nd_array, float
            An array of the optimal weights, which maximize the barrier in the
            cut-based free energy profile.
        """

        def objective(weights, generators):
            """ returns the negative of the MFPT """
            RC = coordinate_fxn(generators, weights)
            zc, zh = calc_cFEP(assignments, lag_time, rxn_coordinate, rescale)
            mfpt = mfpt(reactant, product, rxn_coordinate, zc, zh)
            return -1.0 * mfpt

        optimal_weights = scipy.optimize.fmin( objective, args=generators, tol=0.0001, 
                                               ftol=0.0001, maxiter=None )

        self.rc_weights = optimal_weights

        return optimal_weights


    def reaction_mfpt(dt=1.0):
        """
        mfpt
        """

        const = dt / np.pi

        A = rxn_coordinate[self.reactant]
        B = rxn_coordinate[self.product]

        perm = np.argsort(self.rxn_coordinate)
        start = np.where(self.rxn_coordinate[perm] == A)
        end = np.where(self.rxn_coordinate[perm] == B)

        intermediate_states = [perm][start:end]
        tau = np.sum( [ ( (zh[x]/(zc[x]**2)) * (np.sum( zh[:x] ) / x) ) \
                        for x in intermediate_states ] )

        return const * tau


    def partition_functions(rxn_coordinate, rescale=True):
        """
        Computes the partition function of the cut-based free energy profile based
        on transition network and reaction coordinate. 
   
        Employs an MSM to do so, thus taking input in the form of a discrete state
        space and the observed dynamics in that space (assignments). Further, one
        can provide a "reaction coordiante" who's values will be employed to
        find the min-cut/max-flow coordinate describing dynamics on that space.
   
        Generously contributed by Sergei Krivov
        Optimizations and modifications due to TJL
   
        Parameters
        ----------
        assignments : nd_array, int
            The MSM assignments array, indicating membership of each snapshot
        lag_time : int
            The MSM lag time, in units of frames
        rxn_coordinate : nd_array, float
            An array of the value of the reaction coordiante for each MSM state
   
        Returns
        -------
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
    
        state_order_along_RC = np.argsort(self.rxn_coordinate_values)

        # generate a permutation matrix that will re-order the counts to be
        # indexed in order of the rxn coordinate array
        perm = scipy.sparse.lil_matrix(self.counts.shape)
        for i in range(N):
            perm[i,state_order_along_RC[i]] = 1.0 # send i -> correct place
        perm_counts = perm * counts * perm.T

        # set up the three variables for weave -- i'm not a weave expert, but
        # it seems like they need to be in the function scope before I can pass
        # them into C++
        data = perm_counts.data
        indices = perm_counts.indices
        indptr = perm_counts.indptr

        zc = np.zeros(N)
    
        scipy.weave.inline(r"""
        int i, j, k;
        double nij, incr;
    
        for (i = 0; i < N; i++) {
            for (k = indptr[i]; k < indptr[i+1]; k++) {
                j = indices[k];
                if (i == j) { continue; }
            
                nij = data[k];
            
                // iterate through all values in the matrix in csr format
                // i, j are the row and column indices
                // nij is the entry
            
                incr = j < i ? nij : -nij;
                zc[j] += incr;
                zc[i] -= incr;
            }
        }
        """, ['N', 'data', 'indices', 'indptr', 'zc'])
    
        zc /= 2.0 # we overcounted in the above - fix that
                            
        inv_ordering = np.argsort(state_order_along_RC)
        zc = zc[inv_ordering]
    
        # dont need to apply inv_ordering since we're doing this on
        # counts not perm_counts
        zh = np.array(counts.sum(axis=0)).flatten()
        
        return zc, zh
    
    

    def rescale_to_natural_coordinate(zc, zh, rxn_coordinate):
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
        
        Returns
        -------
        natural_coordinate : ndarray, float
            The reaction coordinate values along the rescaled coordinate
        scaled_z : ndarray, float
            The partition function value for each point of `natural_coordinate`
    
        See Also
        --------
        calc_cFEP
        """
    
        state_order_along_RC = np.argsort(rxn_coordinate)
        zc = zc[state_order_along_RC]
        zh = zh[state_order_along_RC]
    
        scaled_z = np.cumsum(zc)
        positive_inds = np.where(scaled_z > 0.0)
        scaled_z = scaled_z[positive_inds]
        natural_coordinate = np.cumsum( zh[positive_inds] / (scaled_z * np.sqrt(np.pi)) )
    
        return natural_coordinate, scaled_z
        
        
        
    def plot(num_bins=20, filename=None):
        """
        Plot the current cut-based free energy profile.
        
        Can either display this on screen, or 
        
        Parameters
        
        """
        
        if self.zc == None:
            self.
        else:
            pass
        

        print "Plotting reaction coordinate"
        perm = np.argsort(self.rxn_coordinate_values)

        # performing a sliding average
        h = self.rxn_coordinate_values.max() / float(num_bins)
        bins = [ i*h for i in range(num_bins) ]

        inds = np.digitize( self.rxn_coordinate_values, bins )
        pops = [ np.sum( self.zc[np.where( inds == i )] ) for i in range(num_bins) ]

        fig = plt.figure()
        plt.plot(bins, -1.0 * np.log( pops ), lw=2)
        plt.xlabel('Reaction Coordinate')
        plt.ylabel('Free Energy Profile (kT)')
        
        if not filename:
            plt.show()
        else:
            plt.savefig(filename)
            print "Saved reaction coordinate plot to: %s" % filename

        return