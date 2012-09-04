#!/usr/bin/env python
 
#from optparse import OptionParser
#parser = OptionParser()
#parser.add_option('-t',dest='traj_dir',default='./Trajectories',help='Directory to find trajectories [ ./Trajectories ]')
#parser.add_option('-o',dest='outFN',default='PCAData.h5',help='File to save the results to. [ PCAData.h5 ]')
#parser.add_option('--dt',dest='dt',type=int,help='Time to calculate the cross correlations (frames)')
#parser.add_option('-u',dest='stride',default=1,type=int,help='Stride to subsample at to train the PCA for. [ 1 ]')

#options, args = parser.parse_args()
from msmbuilder import arglib, sshfs_tools, tICA
import numpy as np
from schwancrtools import tPCA, dataIO, msmTools
from msmbuilder import Project, Serializer
from schwancrtools.Trajectory_crs import Trajectory
import os, sys, re
import scipy

parser = arglib.ArgumentParser( get_basic_metric = True )
parser.add_argument('project')
parser.add_argument('stride',help='stride to subsample input trajectories',type=int,default=1)
parser.add_argument('atom_indices',help='atom indices to restrict trajectories to',default=None)
parser.add_argument('out_fn',help='output filename to save results to',default='tICAData.h5')
parser.add_argument('delta_time',help='delta time to use in calclating the time-lag correlation matrix',type=int)
parser.add_argument('procs',help='number of processors to use with multiprocessing',type=int,default=1)
parser.add_argument('min_length',help='only train on trajectories greater than some number of frames',type=int,default=0)

args, prep_metric = parser.parse_args()
print args
print prep_metric
exit()
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

