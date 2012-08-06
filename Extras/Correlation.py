# This file is part of MSMBuilder.
#
# Copyright 2011 Stanford University
#
# MSMBuilder is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

import numpy as np
from scipy import signal
from msmbuilder import MSMLib
    
def Correlation(x,Normalize=True,MaxTime=np.inf):
    """Calculate the (auto) correlation function of an sequence of observables x.
    """
    x=x-np.mean(x)
    n=x.shape[0]
    ACF=[np.var(x)]
    for t in xrange(1,min(MaxTime,n)):
        ACF.append(np.dot(x[t:],x[:-t])/(float(n)-float(t)))
    ACF=np.array(ACF)
    if Normalize==True:
        ACF/=ACF[0]
    return(ACF)

def fft_autocorrelate(A):
    A -= A.mean()
    result = signal.fftconvolve(A,A[::-1])
    var = A.std()**2
    return result[result.size/2:] / var

def RawMSMCorrelation(T,ObservableArray,AssignmentArray,Steps=10000,StartingState=0):
    """Calculate an autocorrelation function from an MSM.  Inputs include: Transition Matrix T, an array of the observable calculated at every project frame, and the array of assignments.  This function works by first generating a 'sample' trajectory from the MSM.  This approach is necessary as it allows treatment of intra-state dynamics.
    """
    Traj=MSMLib.Sample(T,StartingState,Steps)
    ObsTraj=[]
    MaxInt = np.ones( AssignmentArray.shape ).sum()
    #for k,State in enumerate(Traj):
    #    Obs=ObservableArray[np.where(AssignmentArray==State)]
    #    MaxNum=len(Obs)
    #    Samp=np.random.random_integers(0,MaxNum-1)
    #    ObsTraj.append(Obs[Samp])
    ObsTraj = np.array( [ ObservableArray[ np.where( AssignmentArray == State ) ].take( [ np.random.randint( MaxInt ) ], mode='wrap' ) for State in Traj ] )

    #ObsTraj=np.array(ObsTraj)
    Cor=fft_autocorrelate( ObsTraj )
    #Cor=Correlation(ObsTraj)
    return(Cor,Traj,ObsTraj)

