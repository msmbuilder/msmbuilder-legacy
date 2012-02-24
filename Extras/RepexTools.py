"""Tools for analyzing Replica Exchange / Dynamical Reweighting Simulations

Notes:

Parts taken from the dynamical reweigthing toolkit.

Please cite the dynamical reweighting paper (X)

"""

import numpy
import scipy.sparse
import datetime # time and date

from msmbuilder import MSMLib

from pymbar import pymbar # for MBAR analysis
import pymbar.timeseries as timeseries # for timeseries analysis

import simtk.unit as units

from RepexHelperFunctions import *

class RepexAnalyzer():

    def Process(self):
        """Process a replica exchange simulation using dynamical reweighting.
        """
        #self.WriteMixingStatistics()
        #self.WriteAcceptanceProbabilities()
        self.EstimateStatisticalInefficiency()
        self.InitializeOutputNetCDF()
        self.CalculatePathHamiltonians()
        self.ComputeReducedPotentials()
        self.CalculateWeights()
        #self.WriteHeatCapacity()
        #self.CloseFiles()

    def WriteMixingStatistics(self,OutFilename="swap-statistics.txt"):
        """Calculate the mixing statistics and write to disk.
        """
        Tij = show_mixing_statistics(self.repex_ncfile, show_transition_matrix=True)
        numpy.savetxt(OutFilename,numpy.array(Tij))

    def WriteAcceptanceProbabilities(self,OutFilename="fraction-accepted.txt"):
        """Show acceptance probabilities, broken down by separation in number of temperatures.
        """
        fraction_accepted=show_acceptance_statistics(self.repex_ncfile)
        numpy.savetxt(OutFilename,numpy.array(fraction_accepted))
    
    def __init__(self,repex_ncfile,output_filename):

        self.repex_ncfile=repex_ncfile
        self.output_filename=output_filename
        print "Reading temperatures and other dimensions from parallel tempering dataset..."
        print "Reading statistics..."
        [self.N,self.K, self.natoms, self.ndim] = repex_ncfile.variables['positions'].shape

        self.T=1
        self.nequil=0
        self.thermodynamic_states=numpy.array([repex_ncfile.variables["states"][:]]).transpose([2,1,0])

        print "%d trajectories" % self.N
        print "%d replicas" % self.K
        print "%d snapshots/trajectory" % self.T
        print "%d atoms in trajectories" % self.natoms
        print "%d dimensions/atom" % self.ndim
        print ""
        

        # Read temperatures.
        if False:#KAB
            self.temperature_k = self.repex_ncfile.variables['temperatures'][:].copy()
        else:
            self.temperature_k=numpy.loadtxt("./Temperatures.dat")
        self.temperature_k = units.Quantity(self.temperature_k, units.kelvin)
        self.beta_k = 1.0 / (kB * self.temperature_k)

    def EstimateStatisticalInefficiency(self):
        """Estimate statistical inefficiency and determine subset of effectively uncorrelated samples.
        """
        # Compute negative log-probability of product space of all replicas.
        print "Computing log-probability history..."
        u_n = numpy.zeros([self.N], numpy.float64)
        for iteration in range(self.N):
           u_n[iteration] = 0.0
           for replica in range(self.K):
              state = self.repex_ncfile.variables['states'][iteration,replica]
              u_n[iteration] += self.repex_ncfile.variables['energies'][iteration,replica,state]

        # Compute statistical inefficiency.
        print "Estimating statistical inefficiency after discarding first %d iterations to equilibration" % self.nequil
        g_u = timeseries.statisticalInefficiency(u_n[self.nequil:])
        print "g_u = %8.1f iterations" % g_u

        # Determine indices of effectively uncorrelated trajectories.
        self.indices = timeseries.subsampleCorrelatedData(u_n[self.nequil:], g=g_u)
        self.indices = numpy.array(self.indices) + self.nequil

        #DEBUG!!!!!!!!!!!!!!!!!!!!!!1
        self.indices=numpy.arange(self.N)

        # Reduce number of samples.
        self.N_Reduced = self.indices.size
        print "There are %d uncorrelated samples (after discarding initial %d and subsampling by %.1f)" % (self.N, self.nequil, g_u)

    def InitializeOutputNetCDF(self):
        """Initialize the NetCDF file for output of computed data objects.
        """
        print "Opening analysis NetCDF file for writing..."
        self.output_ncfile = netcdf.Dataset(self.output_filename, 'w', format='NETCDF3_CLASSIC')

        # Set global attributes.
        setattr(self.output_ncfile, 'title', "Analysis data produced at %s" % datetime.datetime.now().ctime())
        setattr(self.output_ncfile, 'application', 'analyze-correlation-function.py')

        # Store dimensions in netcdf.
        self.output_ncfile.createDimension('K', self.K)             # number of temperatures
        self.output_ncfile.createDimension('N', self.N)             # number of trajectories per temperature
        self.output_ncfile.createDimension('T', self.T)             # number of snapshots per trajectory

        variable = self.output_ncfile.createVariable('temperature_k', 'd', ('K',))
        setattr(variable, 'units', 'Kelvin')
        setattr(variable, 'description', 'temperature_k[k] is the temperature of temperature index k')
        self.output_ncfile.variables['temperature_k'][:] = self.temperature_k / units.kelvin

        variable = self.output_ncfile.createVariable('beta_k', 'f', ('K',))
        setattr(variable, 'units', '1/(kcal/mol)')
        setattr(variable, 'description', 'beta_k[k] is the inverse temperature of temperature index k')
        self.output_ncfile.variables['beta_k'][:] = self.beta_k / (1.0 / units.kilocalories_per_mole)

    def CalculatePathHamiltonians(self):
        """Read path Hamiltonians for uncorrelated trajectories.
        """
        print "Computing path Hamiltonians..."
        self.H_kn = units.Quantity(numpy.zeros([self.K,self.N], numpy.float64), units.kilocalories_per_mole)
        for n in range(self.N):
           # Get index into original iteration.
           iteration = self.indices[n]
           

           # Compute path Hamiltonians.
           for replica in range(self.K):
              state = self.repex_ncfile.variables['states'][iteration,replica]
              u = float(self.repex_ncfile.variables['energies'][iteration,replica,state])
              self.H_kn[state,n] = u / self.beta_k[state]

        variable = self.output_ncfile.createVariable('H_kn', 'd', ('K','N'))
        setattr(variable, 'units', 'kcal/mol')
        setattr(variable, 'description', 'H_kn[k,n] is the path Hamiltonian of trajectory n from state k')
        self.output_ncfile.variables['H_kn'][:,:] = self.H_kn[:,:] / units.kilocalories_per_mole

    def ComputeReducedPotentials(self):
        """Compute reduced potentials in all states for MBAR.
        """
        print "Computing reduced potentials..."
        self.u_kln = numpy.zeros([self.K,self.K,self.N], numpy.float64)
        for frame in range(self.N):
           for k in range(self.K):
               self.u_kln[k,:,frame] = self.beta_k[:] * self.H_kn[k,frame]

        #v_kln are the normal (non dynamical) reduced potentials
        self.v_kln=self.repex_ncfile.variables['energies'][:].copy().transpose((1,2,0))
        self.u_kln=self.repex_ncfile.variables['energies'][:].copy().transpose((1,2,0))

    def CalculateWeights(self):
        """Use MBAR to calculate thermodynamic weights.
        """
        #===================================================================================================
        # Initialize MBAR.
        #===================================================================================================

        print "Initiaizing MBAR..."
        self.N_k = self.N * numpy.ones([self.K], numpy.int32) # N_k[k] is the number of uncorrelated samples from themodynamic state k
        self.mbar = pymbar.MBAR(self.u_kln, self.N_k, verbose=True, method='Newton-Raphson', initialize='BAR', relative_tolerance = 1.0e-10)

        #===================================================================================================
        # Compute weights at each temperature.
        #===================================================================================================

        # Choose temperatures for reweighting to be the simulation temperatures plus their midpoints.
        self.L = 2*self.K-1 # number of temperatures for reweighting
        self.reweighted_temperature_l = units.Quantity(numpy.zeros([self.L], numpy.float64), units.kelvin)
        for k in range(self.K):
           self.reweighted_temperature_l[2*k] = self.temperature_k[k] # simulation temperatures
        for k in range(self.K-1):   
           self.reweighted_temperature_l[2*k+1] = (self.temperature_k[k] + self.temperature_k[k+1]) / 2.0 # midpoint temperatures
        print "Temperatures for reweighting:"
        print self.reweighted_temperature_l

        print "Computing trajectory weights for reweighting temperatures..."
        self.log_w_lkn = numpy.zeros([self.L,self.K,self.N], numpy.float64) # w_lkn[l,k,n] is the normalized weight of snapshot n from simulation k at reweighted temperature l
        self.w_lkn = numpy.zeros([self.L,self.K,self.N], numpy.float64) # w_lkn[l,k,n] is the normalized weight of snapshot n from simulation k at reweighted temperature l

        # alternate: first compute just denominators    
        all_log_denom = self.mbar._computeUnnormalizedLogWeights(numpy.zeros([self.mbar.K,self.mbar.N_max],dtype=numpy.float64))
        for l in range(self.L):
           temperature = self.reweighted_temperature_l[l]
           beta = 1.0 / (kB * temperature)
           u_kn = beta * self.H_kn
           log_w_kn = -u_kn+all_log_denom
           w_kn = numpy.exp(log_w_kn - log_w_kn.max())
           w_kn = w_kn / w_kn.sum()
           self.w_lkn[l,:,:] = w_kn
           self.log_w_lkn[l,:,:] = log_w_kn   

        # Store weights.
        self.output_ncfile.createDimension('L', self.L)             # number of temperatures for reweighting

        variable = self.output_ncfile.createVariable('reweighted_temperature_l', 'd', ('L',))
        setattr(variable, 'units', 'Kelvin')
        setattr(variable, 'description', 'reweighted_temperature_l[k] is the temperature of reweighted temperature index k')
        self.output_ncfile.variables['reweighted_temperature_l'][:] = self.reweighted_temperature_l / units.kelvin

        variable = self.output_ncfile.createVariable('log_w_lkn', 'd', ('L','K','N'))
        setattr(variable, 'units', 'dimensionless')
        setattr(variable, 'description', 'log_w_lkn[l,k,n] is the unnormalized log weight of trajectory n from temperature k at reweighted temperature l')
        self.output_ncfile.variables['log_w_lkn'][:,:,:] = self.log_w_lkn

        variable = self.output_ncfile.createVariable('w_lkn', 'd', ('L','K','N'))
        setattr(variable, 'units', 'dimensionless')
        setattr(variable, 'description', 'w_lkn[l,k,n] is the normalized weight of trajectory n from temperature k at reweighted temperature l')
        self.output_ncfile.variables['w_lkn'][:,:,:] = self.w_lkn

    def WriteHeatCapacity(self,OutFilename="generalized-head-capacity.txt"):
        """Compute generalized heat capacity as a function of temperature.
        """
        print "Computing generalized heat capacity as a function of temperature..."
        heat_capacity_units = units.kilocalories_per_mole / units.kelvin
        Cv_l = units.Quantity(numpy.zeros([self.L], numpy.float64), heat_capacity_units)
        for l in range(self.L):
           temperature = self.reweighted_temperature_l[l]
           beta = 1.0 / (kB * temperature)
           EH = numpy.sum(self.w_lkn[l,:,:] * (self.H_kn/self.H_kn.unit)) * self.H_kn.unit
           EH2 = numpy.sum(self.w_lkn[l,:,:] * (self.H_kn/self.H_kn.unit)**2) * self.H_kn.unit**2
           varH = numpy.sum(self.w_lkn[l,:,:] * ((self.H_kn - EH)/self.H_kn.unit)**2) * self.H_kn.unit**2
           Cv_l[l] = kB * beta**2 * varH

        variable = self.output_ncfile.createVariable('Cv_l', 'd', ('L',))
        
        setattr(variable, 'units', 'kcal/mol/kelvin')
        setattr(variable, 'description', 'Cv_l[l] is the generalized heat capacity of reweighted temperature l')
        self.output_ncfile.variables['Cv_l'][:] = Cv_l[:] / heat_capacity_units

        outfile = open(OutFilename,'w')
        for l in range(self.L):
            outfile.write('%8.1f K : %16.3f kcal/mol/K' % (self.reweighted_temperature_l[l] / units.kelvin, Cv_l[l] / heat_capacity_units))
        outfile.close()

    def CloseFiles(self):
        """Close all open files.
        """
        self.repex_ncfile.close()
        self.output_ncfile.close()
        print "Done."

    def ConvertShapeToRepex(self,Observable):
        """Convert shapes.

        Notes:
        The shape (self.T) is currently hardcoded for single frame replicas.
        """
        
        NewObservable=numpy.zeros((self.K,self.N,self.T))
        for replica in range(self.K):
            NewObservable[replica,:,0]=Observable[replica]
            
        return NewObservable

    def LoadStateAssignments(self,Assignments):
        Assigned=self.ConvertShapeToRepex(Assignments)
        
        self.state_knt=numpy.zeros((self.K,self.N,self.T+1),'int')
        self.state_knt[:,:,:-1]=Assigned
        for i in range(self.N-1):
            for j in range(self.K):
                Ind=numpy.where(self.thermodynamic_states[:,i+1,0]==self.thermodynamic_states[j,i,0])
                self.state_knt[j,i,-1]=self.state_knt[Ind,i+1,0]

    def PrepareCountMatrix(self,tau=1):
        """Compute correlation functions at individual temperatures.

        Arguments:
        """
        NumStates=self.state_knt.max()+1
        self.NumStates=NumStates
        AllCounts=[]
        for i, Temperature in enumerate(xrange(self.K)):
            SingleTemperatureAssignments=self.state_knt[self.thermodynamic_states[:,:,0]==Temperature].transpose()
            AllCounts.append(MSMLib.GetCountMatrixFromAssignments(SingleTemperatureAssignments,NumStates=NumStates))
            if i==0:
                self.CountsOverAllTemperatures=scipy.sparse.lil_matrix(AllCounts[0].shape)
            self.CountsOverAllTemperatures+=AllCounts[i]
        self.AllCounts=AllCounts
        self.CountsOverAllTemperatures=self.CountsOverAllTemperatures.tocsr()

        #We don't want intermediate temperatures here, so subsample over them
        Weights=self.w_lkn[::2].copy()

        #Nonzero entries of count matrix over all temperatures
        NonZero=numpy.array(self.CountsOverAllTemperatures.nonzero()).transpose()

        s=self.state_knt#Define for syntax ease

        print "Computing observables Akn..."

        """
        #Note: Forming elements into observables is NOT necessary because we already have 1D structure due to sparse CSR format
        A_ijkn = numpy.zeros([len(NonZero), self.K, self.N], 'float')
        for Frame in range(self.N):
            for Temperature in range(self.K):
                C=MSMLib.GetCountMatrixFromAssignments(s[:,Frame,:],NumStates=NumStates)
                C=0.5*(C+C.transpose())

                for k,(i,j) in enumerate(NonZero):
                    A_ijkn[k,Temperature,Frame]=C[i,j]
                

        """

        A_ijkn = numpy.zeros([NumStates,NumStates, self.K, self.N], 'float')
        T=self.T
        for i in xrange(NumStates):
            for j in xrange(NumStates):
                A_ijkn[i,j,:,:] = numpy.mean((s[:,:,0:(1+T-tau)]==i) & (s[:,:,tau:(T+1)]==j) | (s[:,:,0:(T-tau+1)]==j) & (s[:,:,tau:(T+1)]==i), axis=2)
                if (i != j): A_ijkn[i,j,:,:] /= 2

        self.A_ijkn=A_ijkn
                
    def ComputeCountMatrix(self,Sparse=False,tau=1):
        try:
            self.A_ijkn
        except:
            self.PrepareCountMatrix(tau=tau)
        Ck=numpy.zeros((self.K,self.NumStates,self.NumStates))
        Ek=numpy.zeros((self.K,self.NumStates,self.NumStates))
        if Sparse==True:
            return
        else:
            for a in range(self.NumStates):
                for b in range(self.NumStates):
                    x=self.mbar.computeExpectations(self.A_ijkn[a,b])
                    Ck[:,a,b]=x[0]
                    Ek[:,a,b]=x[1]
        return Ck
        



    def SetupStaticMBar(self):
        """Prepares an MBar calculation for non-dynamical calculations (e.g. NOT using dynamical reweighting)."""

        print "Computing reduced potential energies..."
        N_k = self.N * numpy.ones([self.K], numpy.int32) # N_k[k] is the number of uncorrelated samples from themodynamic state k
        self.mbar2 = pymbar.MBAR(self.v_kln, N_k, verbose=True, method='Newton-Raphson', initialize='BAR', relative_tolerance = 1.0e-10)
         
