#!/usr/bin/python

# Process trajectory data to compute torsions and weights.

#=============================================================================================
# REQUIREMENTS
#
# This code requires the 'pynetcdf' package, containing the Scientific.IO.NetCDF package built for numpy.
#
# http://pypi.python.org/pypi/pynetcdf/
# http://sourceforge.net/project/showfiles.php?group_id=1315&package_id=185504
#
# This code also uses the 'MBAR' package, implementing the multistate Bennett acceptance ratio estimator, available here:
#
# http://www.simtk.org/home/pymbar
#=============================================================================================

import numpy
import numpy.linalg 

import simtk.unit as units

import netCDF4 as netcdf # for writing of data objects for plotting in Matlab or Mathematica

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant in energy/temperature units

def logSum(log_terms):
   """Compute the log of a sum of terms whose logarithms are provided.

   REQUIRED ARGUMENTS  
      log_terms is the array (possibly multidimensional) containing the logs of the terms to be summed.

   RETURN VALUES
      log_sum is the log of the sum of the terms.

   """

   # compute the maximum argument
   max_log_term = log_terms.max()

   # compute the reduced terms
   terms = numpy.exp(log_terms - max_log_term)

   # compute the log sum
   log_sum = log( terms.sum() ) + max_log_term

   # return the log sum
   return log_sum

def show_mixing_statistics(ncfile, show_transition_matrix=False):
    """
    Compute mixing statistics among thermodynamic states.

    OPTIONAL ARGUMENTS

    show_transition_matrix (boolean) - if True, the transition matrix will be printed

    RETURN VALUES

    Tij (numpy array of dimension [nstates,nstates]) - Tij[i,j] is the fraction of time a pair of replicas at states i and j were swapped during an iteration    

    """

    print "Computing mixing statistics..."

    states = ncfile.variables['states'][:,:].copy()

    # Determine number of iterations and states.
    [niterations, nstates] = ncfile.variables['states'][:,:].shape
    
    # Compute statistics of transitions.
    Nij = numpy.zeros([nstates,nstates], numpy.float64)
    for iteration in range(niterations-1):
        for ireplica in range(nstates):
            istate = states[iteration,ireplica]
            jstate = states[iteration+1,ireplica]
            Nij[istate,jstate] += 0.5
            Nij[jstate,istate] += 0.5
    Tij = numpy.zeros([nstates,nstates], numpy.float64)
    for istate in range(nstates):
        Tij[istate,:] = Nij[istate,:] / Nij[istate,:].sum()

    if show_transition_matrix:
        # Print observed transition probabilities.
        PRINT_CUTOFF = 0.001 # Cutoff for displaying fraction of accepted swaps.
        print "Cumulative symmetrized state mixing transition matrix:"
        print "%6s" % "",
        for jstate in range(nstates):
            print "%6d" % jstate,
        print ""
        for istate in range(nstates):
            print "%-6d" % istate,
            for jstate in range(nstates):
                P = Tij[istate,jstate]
                if (P >= PRINT_CUTOFF):
                    print "%6.3f" % P,
                else:
                    print "%6s" % "",
            print ""

    # Estimate second eigenvalue and equilibration time.
    mu = numpy.linalg.eigvals(Tij)
    mu = -numpy.sort(-mu) # sort in descending order
    if (mu[1] >= 1):
        print "Perron eigenvalue is unity; Markov chain is decomposable."
    else:
        print "Perron eigenvalue is %9.5f; state equilibration timescale is ~ %.1f iterations" % (mu[1], 1.0 / (1.0 - mu[1]))

    return Tij

def show_acceptance_statistics(ncfile):
    """
    Print summary of exchange acceptance statistics.

    ARGUMENTS
       ncfile (NetCDF file handle) - the parallel tempering datafile to be analyzed

    RETURNS
       fraction_accepted (numpy array of [nstates-1]) - fraction_accepted[separation] is fraction of swaps attempted between state i and i+separation that were accepted

    """

    print "Computing acceptance statistics..."

    nstates = ncfile.variables['proposed'][:,:,:].shape[1]

    # Aggregated proposed and accepted by how far from the diagonal we are.
    fraction_accepted = numpy.ones([nstates-1], numpy.float64)
    for separation in range(1,nstates-1):
        nproposed = 0
        naccepted = 0            
        for i in range(nstates):
            j = i+separation
            if (j < nstates):                
                nproposed += ncfile.variables['proposed'][:,i,j].sum()
                naccepted += ncfile.variables['accepted'][:,i,j].sum()
        fraction_accepted[separation] = float(naccepted) / float(nproposed)
        print "%5d : %10d %10d : %8.5f" % (separation, nproposed, naccepted, fraction_accepted[separation])

    return fraction_accepted
