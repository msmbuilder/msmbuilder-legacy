#!/usr/bin/env python
"""Interactively estimate a rate matrix usign SCRE.
"""

import os, sys
import scipy.io
from mdtraj import io
from msmbuilder import MSMLib, SCRE, arglib
import numpy as np
import string
import matplotlib
import logging
logger = logging.getLogger('msmbuilder.scripts.Interactive-SCRE')

parser = arglib.ArgumentParser(description=__doc__)
parser.add_argument('output_dir')
parser.add_argument('assignments')


def interactive_scre(assignments):
    Counts = MSMLib.get_count_matrix_from_assignments(assignments, lag_time=1)
    CountsAfterTrimming,Mapping = MSMLib.ErgodicTrim(Counts)
    MSMLib.ApplyMappingToAssignments(assignments,Mapping)
    ReversibleCounts = MSMLib.EstimateReversibleCountMatrix(CountsAfterTrimming)
    T = MSMLib.EstimateTransitionMatrix(ReversibleCounts).toarray()
    populations = np.array(ReversibleCounts.sum(0)).flatten()
    populations /= populations.sum()

    K0=SCRE.ConvertTIntoK(T)
    M,X=SCRE.get_parameter_mapping(K0)

    while len(X) > 0:
        lagtime_list = get_lagtimes()
        KList = scre_iteration(assignments,K0,lagtime_list,M,X,populations)
        matplotlib.pyplot.show()
        if len(X) > 1:
            i,j,lagtime = get_input()       
            SCRE.FixEntry(M,X,populations,K0,i,j,KList[lagtime][i,j])
        else:
            lagtime = get_lagtime()
            return KList[lagtime]


def get_lagtime():
    try:
        lagtime = int(raw_input("Enter a lagtime (int), in units of the data storage step.\n"))
    except ValueError:
        lagtimes = get_lagtimes()
    return lagtime - 1
        
def get_lagtimes():
    try:
        max_lagtime = int(raw_input("Enter a maximum lagtime (int), in units of the data storage step.\n"))
        lagtimes = np.arange(1,max_lagtime)
    except ValueError:
        lagtimes = get_lagtimes()

    return lagtimes
    

def get_input():
    try:
        s = raw_input("Enter i,j,lagtime (int, int, int), where  lagtime is in units of the data storage step.\n")
        i,j,lagtime = np.array(string.split(s,','),'int')
    except ValueError:
        i,j,lagtime = get_input()

    return i,j,lagtime - 1 

def scre_iteration(assignments,K0,lagtime_list,M,X,populations):    
    KList=[]
    counts_list = []
    for LagTime in lagtime_list:
        logger.info("Estimating rates at lagtime %d", LagTime)
        K=K0.copy() * float(LagTime)
        C0=MSMLib.get_count_matrix_from_assignments(assignments, lag_time=LagTime).toarray()
        Counts=C0.sum(1)
        Counts/=LagTime     
        X2=SCRE.MaximizeRateLikelihood(X,M,populations,C0,K)
        K=SCRE.ConstructRateFromParams(X2,M,populations,K)
        K/=(LagTime)
        KList.append(K)
        counts_list.append(Counts)

    KList=np.array(KList)
    SCRE.PlotRates(KList,lagtime_list,counts_list)
    return KList

run = interactive_scre

if __name__ == "__main__":
    args = parser.parse_args()

    try:
        assignments = io.loadh(args.assignments, 'arr_0')
    except KeyError:
        assignments = io.loadh(args.assignments, 'Data')

    K = run(assignments)

    T = scipy.linalg.matfuncs.expm(K)

    np.savetxt(os.path.join(args.output_dir, "Rate.dat"), K)
    scipy.io.mmwrite(os.path.join(args.output_dir, "tProb.mtx.tl"), T)
