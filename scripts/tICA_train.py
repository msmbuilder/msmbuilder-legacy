#!/usr/bin/env python
 
#from optparse import OptionParser
#parser = OptionParser()
#parser.add_option('-t',dest='traj_dir',default='./Trajectories',help='Directory to find trajectories [ ./Trajectories ]')
#parser.add_option('-o',dest='outFN',default='PCAData.h5',help='File to save the results to. [ PCAData.h5 ]')
#parser.add_option('--dt',dest='dt',type=int,help='Time to calculate the cross correlations (frames)')
#parser.add_option('-u',dest='stride',default=1,type=int,help='Stride to subsample at to train the PCA for. [ 1 ]')

#options, args = parser.parse_args()
from schwancrtools import ArgLib_E, sshfs_tools
import numpy as np
from schwancrtools import tPCA, dataIO, msmTools
from msmbuilder import Project, Serializer
from schwancrtools.Trajectory_crs import Trajectory
import os, sys, re
import scipy
 
options, prep_metric = ArgLib_E.parse(['projectfn','stride','atomindices'],
     new_arglist=[('outFN','--out','--outputFN',"Output filename to save the results to",'PCAData.h5'),
                  ('dt','--dt','--delta-time',"Time to calcualte the cross correlations (frames)",None),
                  ('procs','-P','--procs',"Processors to use in calculating the correlations.",1),
                  ('minlength','--ml','--min-length',"Minimum length a trajectory needs to be in order to use it.",0),],
      metric_parsers=True)

print prep_metric

if os.path.exists( options.outFN ):
    print "Out filename (%s) already exists..." % options.outFN
    exit()

N = 0 # This tracks the number of observations

procs = int( options.procs )
try: aind = dataIO.readData( options.atomindices ).astype(int)
except: aind = None
stride = int( options.stride )
dt = int( options.dt )
Proj = Project.LoadFromHDF(options.projectfn)
minlength = int( float( options.minlength ) ) # need to convert to float first because int can't convert a string that is '1E3' for example...wierd.
lag = int( dt / stride )

if float(dt)/stride != lag:
    print "Stride must be a divisor of dt..."
    sys.exit()

if lag > 0:
   cov_mat_obj = tPCA.CovarianceMatrix( lag=lag, procs=procs, normalize=True )
else:
   cov_mat_obj = tPCA.CovarianceMatrix( lag=lag, procs=procs, normalize=False )


for i in xrange(Proj['NumTrajs']):
    print "Working on trajectory %d" % i
    if Proj['TrajLengths'][i] <= lag:
        print "\tTrajectory is not long enough for this lag (%d vs %d)" % ( Proj['TrajLengths'][i], lag )
        continue
    if Proj['TrajLengths'][i] < minlength:
        print "\tTrajectory is not longer than minlength (%d vs %d)" % ( Proj['TrajLengths'][i], minlength )
        continue
#    traj = Proj.LoadTraj(i)
    traj = Trajectory.LoadFromLHDF( Proj.GetTrajFilename( i ), Stride=stride, AtomIndices=aind )
    ptraj = prep_metric.prepare_trajectory( traj )
    del traj
    n_cols = ptraj.shape[1]
    if cov_mat_obj.size == None:
        cov_mat_obj.set_size( n_cols )

    print "here"
    cov_mat_obj.train(ptraj)
    del ptraj
    print "here"
    if ( not (i+1) % 10 ):
        print "Remounting..."
        sshfs_tools.remount()

sshfs_tools.remount()

print "Diagonalizing the covariance matrix"

if lag > 0:
    cov_mat, cov_mat_lag0 = cov_mat_obj.get_current_estimate()
    vals, vecs = scipy.linalg.eigh( cov_mat, b=cov_mat_lag0 ) # Get the right eigenvectors of the covariance matrix. It's hermitian so left=right e-vectors
else:
    cov_mat = cov_mat_obj.get_current_estimate()
    vals, vecs = scipy.linalg.eigh( cov_mat ) # Get the right eigenvectors of the covariance matrix. It's hermitian so left=right e-vectors

print "Saving output to %s" % options.outFN 
#out_serializer = Serializer( { 'r_vecs' : r_vecs, 'l_vecs' : l_vecs, 'vals' : vals } )
out_serializer = Serializer( {'vecs' : vecs, 'vals' : vals } ) # eigenvectors are in the columns
out_serializer.SaveToHDF( options.outFN )

