
from scipy import signal
import numpy as np
import re, sys, os
import multiprocessing as mp
from time import time
import gc

def np_dot_row( args ):
    row_ind = args[0]
    lag = args[1]
    sol = []

    if lag == 0:
        a = data_vector[:,row_ind].reshape( (-1,1) )
        sol = (a * data_vector ).sum(axis=0)
    else:
        a = data_vector[:-lag,row_ind].reshape( (-1,1) )
        sol = (a * data_vector[lag:]).sum(axis=0)

    del a
    return sol

def np_correlate_row( args ):
    row_ind = args[0]
    lag = args[1]
    sol = []

    if lag == 0:
        a = data_vector[:,row_ind]
        for j in xrange( data_vector.shape[1] ):
            sol.append( np.correlate( a, data_vector[:,j],mode='valid' )[0] )
    else:
        a = data_vector[:-lag,row_ind]
        for j in xrange( data_vector.shape[1] ):
            sol.append( np.correlate( a, data_vector[lag:,j], mode='valid' )[0] ) 
    del a
    return sol

class CovarianceMatrix:

    def __init__( self, lag=0, procs=1, normalize=False ):

        self.corrs = None
        self.left_sum = None
        self.right_sum = None
        self.tot_sum = None

        self.coors_lag0 = None        
        self.normalize = normalize

        self.trained_frames=0
        self.total_frames = 0

        self.lag=int(lag)

        self.size=None

        self.procs=procs

    def set_size( self, N ):
        if self.corrs !=None:
            print "There is still a matrix stored! Not overwriting, use method start_over() to delete the old matrix."
            return
        self.size = N

        self.corrs = np.zeros( (N,N) )
        self.left_sum = np.zeros( N )
        self.right_sum = np.zeros( N )
        self.tot_sum = np.zeros( N )

        if self.normalize:
            self.corrs_lag0 = np.zeros( (N, N) )

    def start_over( self ):
        self.trained_frames = 0
        self.total_frames = 0

        self.corrs = None
        self.left_sum = None
        self.right_sum = None
        self.tot_sum = None
#        self.sq_tot_sum = None

        self.corrs_lag0 = None

    def train( self, data_vector_orig ):
        global data_vector
        a=time()
        data_vector = data_vector_orig.copy()
        #data_vector /= data_vector.std(axis=0)
        if data_vector.shape[1] != self.size:
            raise Exception("Input vector is not the right size. axis=1 should be length %d. Vector has shape %s" %(self.size, str(data_vector.shape)) )

        if data_vector.shape[0] <= self.lag:
            print "Data vector is too short (%d) for this lag (%d)" % (data_vector.shape[0],self.lag)
            return

        temp_mat = np.zeros( (self.size,self.size) )

        num_frames = data_vector.shape[0] - self.lag
        b=time()
        Pool = mp.Pool( self.procs )
        #sol = [ np_dot_row( (i,self.lag) ) for i in xrange( self.size ) ]
        #debug for memory leak ^^^

        result = Pool.map_async( np_dot_row, zip( range( self.size ), [self.lag]*self.size ) )
        result.wait()
        sol=result.get()

        Pool.close()
        Pool.join()
        temp_mat = np.vstack( sol )

        if self.normalize:
  
            Pool = mp.Pool( self.procs )

            result_lag0 = Pool.map_async( np_dot_row, zip( range( self.size ), [0]*self.size ) )
            result_lag0.wait()
            sol=result_lag0.get()

            Pool.close()
            Pool.join()
            temp_mat_lag0 = np.vstack( sol )

            self.corrs_lag0 += temp_mat_lag0

        c=time()

        self.corrs += temp_mat
        self.left_sum += data_vector[: -self.lag].sum(axis=0)
        self.right_sum += data_vector[ self.lag :].sum(axis=0)
        self.tot_sum += data_vector.sum(axis=0)
        # self.sq_tot_sum += (data_vector * data_vector).sum(axis=0)
        
        self.trained_frames += num_frames
        self.total_frames += data_vector.shape[0]
        f=time()

        #print np.abs(self.corrs_lag0 - self.corrs_lag0.T).max()
        print "Setup: %f, Corrs: %f, Finish: %f" %( b-a, c-b, f-c)

    def get_current_estimate(self):

        tot_means = ( self.tot_sum / float( self.total_frames ) ).reshape( (-1,1) )
 
        tot_means_mat = np.dot( tot_means, tot_means.T ) * self.trained_frames

        left_sum = self.left_sum.reshape( (-1,1) )
        right_sum = self.right_sum.reshape( (-1,1) )

        tot_mean_left_sum = np.dot( tot_means, left_sum.T )
        tot_mean_right_sum = np.dot( tot_means, right_sum.T )



        temp_mat = self.corrs - tot_mean_left_sum - tot_mean_right_sum + tot_means_mat
        temp_mat /= self.total_frames

        temp_mat = (temp_mat+temp_mat.T)/2.

        if self.normalize:

            print np.abs(self.corrs_lag0 - self.corrs_lag0.T).max()
            temp_mat_lag0 = self.corrs_lag0 / float( self.total_frames ) - np.dot( tot_means, tot_means.T )
            return temp_mat, temp_mat_lag0

        return temp_mat
