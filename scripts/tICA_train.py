#!/usr/bin/env python
 
from msmbuilder import arglib, sshfs_tools, tICA
from msmbuilder import Project, Trajectory, io
import numpy as np
import os, sys, re
import scipy
import logging
logger = logging.getLogger( __name__ )

def run( prep_metric, project, atom_indices, out_fn, dt, min_length, lag ):

    if lag > 0: # Then we're doing tICA
        cov_mat_obj = tICA.CovarianceMatrix( lag=lag, calc_cov_mat=True )
    else: # If lag is zero, this is equivalent to regular PCA
        cov_mat_obj = tICA.CovarianceMatrix( lag=lag, calc_cov_mat=False )
    
    for i in xrange( project.n_trajs ):
        logger.info( "Working on trajectory %d" % i )

        if project.traj_lengths[i] <= lag:
            logger.info( "\tTrajectory is not long enough for this lag (%d vs %d)" % ( project.traj_lengths[i], lag ) )
            continue

        if project.traj_lengths[i] < min_length:
            logger.info( "\tTrajectory is not longer than minlength (%d vs %d)" % ( project.traj_lengths[i], min_length ) )
            continue

        traj = Trajectory.load_from_lhdf( project.traj_filename( i ), Stride=stride, AtomIndices=atom_indices )
        ptraj = prep_metric.prepare_trajectory( traj )

        n_cols = ptraj.shape[1]

        cov_mat_obj.train(ptraj)

    logger.info( "Diagonalizing the covariance matrix" )
    
    if lag > 0:
        cov_mat, cov_mat_lag0 = cov_mat_obj.get_current_estimate()
        vals, vecs = scipy.linalg.eig( cov_mat, b=cov_mat_lag0 ) 
        # Note that we can't use eigh because b is positive SEMI-definite, but it would need to be positive definite...
        
        if np.abs( vals.imag ).max() > 1E-8:
            logger.warn( print "Non-real eigenvalues!!!! Something is probably wrong, but I will save everything anyway..." )
        else:
            vals = vals.real
    
        inc_vals_ind = np.argsort( vals ) # Note if these are complex then it will probably be the absolute value that's sorted
        # But if they are complex there are other issues...

        vec_norms = np.array([ vec.dot( cov_mat ).dot( vec ) / val for vec,val in zip(vecs.T, vals) ]) # get the variances for each projection
        vecs /= np.sqrt( vec_norms )
        io.saveh( out_fn, vecs=vecs, vals=vals, cov_mat=cov_mat, cov_mat_lag0=cov_mat_lag0 )
    else:
        cov_mat = cov_mat_obj.get_current_estimate()
        vals, vecs = scipy.linalg.eigh( cov_mat ) # Get the right eigenvectors of the covariance matrix. It's hermitian so left=right e-vectors

        io.saveh( out_fn, vecs=vecs, vals=vals, cov_mat=cov_mat )

    print logger.info( "Saved output to %s" % out_fn )

    return

if __name__ == '__main__':
    parser = arglib.ArgumentParser( get_basic_metric = True )
    parser.add_argument('project')
    parser.add_argument('stride',help='stride to subsample input trajectories',type=int,default=1)
    parser.add_argument('atom_indices',help='atom indices to restrict trajectories to',default='all')
    parser.add_argument('out_fn',help='output filename to save results to',default='tICAData.h5')
    parser.add_argument('delta_time',help='delta time to use in calclating the time-lag correlation matrix',type=int)
    parser.add_argument('min_length',help='only train on trajectories greater than some number of frames',type=int,default=0)
    
    args, prep_metric = parser.parse_args()
    
    arglib.die_if_path_exists( args.out_fn )
    
    try: 
        atom_indices = np.loadtxt( args.atom_indices ).astype(int)
    except: 
        atom_indices = None

    stride = int( args.stride )
    dt = int( args.delta_time )
    project = Project.load_from( args.project )
    min_length = int( float( args.min_length ) ) # need to convert to float first because int can't convert a string that is '1E3' for example...wierd.
    lag = int( dt / stride )

    if float(dt)/stride != lag:
        logger.error( "Stride must be a divisor of dt..." )
        sys.exit()

    run( prep_metric, project, atom_indices, args.out_fn, dt, min_length, lag )



