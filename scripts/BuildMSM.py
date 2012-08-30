#!/usr/bin/python
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

import sys, os
import numpy as np
import scipy.io
from msmbuilder import arglib
from msmbuilder import Serializer
from msmbuilder import MSMLib

# def EstimateUnSym(Counts,Assignments):
#     """Implements the following protocol:
#     1.  Use Tarjan's algorithm to find maximal (strongly) ergodic subgraph.
#     2.  Estimate a general (non-reversible) transition matrix.
#     3.  Calculate populations from stationary eigenvector.
#     """
#     print "Doing no symmetrization. Warning: the resulting model may not satisfy detailed balance and could have complex eigenvalues."
#     CountsAfterTrimming, Mapping = MSMLib.ErgodicTrim(Counts)
#     ReversibleCounts = CountsAfterTrimming
#     MSMLib.ApplyMappingToAssignments(Assignments,Mapping)
#     TC = MSMLib.EstimateTransitionMatrix(ReversibleCounts)
#     EigAns = MSMLib.GetEigenvectors(TC,5)
#     Populations = EigAns[1][:,0]
#     return (CountsAfterTrimming, ReversibleCounts, TC, Populations, Mapping)
# 
# def EstimateSym(Counts,Assignments):
#     """Implements the following protocol:
#     1.  Symmetrize counts via C' = C+C.transpose()
#     2.  Use Tarjan's algorithm to find maximal (strongly) ergodic subgraph.
#     3.  Estimate a reversible transition matrix.
#     4.  Calculate populations from normalized row sums of count matrix.
#     """
#     Counts = 0.5*(Counts + Counts.transpose())
#     ReversibleCounts,Mapping = MSMLib.ErgodicTrim(Counts)
#     MSMLib.ApplyMappingToAssignments(Assignments,Mapping)
#     TC = MSMLib.EstimateTransitionMatrix(ReversibleCounts)
#     Populations = np.array(ReversibleCounts.sum(0)).flatten()
#     Populations /= Populations.sum()
#     CountsAfterTrimming = ReversibleCounts
#     return (CountsAfterTrimming, ReversibleCounts, TC, Populations, Mapping)
# 
# def EstimateMLE(Counts,Assignments,Prior=0.):
#     """Implements the following protocol:
#     1.  Use Tarjan's algorithm to find maximal (strongly) ergodic subgraph.
#     2.  Estimate (via MLE) a reversible transition (TC) and count matrix (ReversibleCounts).
#     3.  Calculate populations from row sums of count matrix.
#     """
#     CountsAfterTrimming,Mapping = MSMLib.ErgodicTrim(Counts)
#     MSMLib.ApplyMappingToAssignments(Assignments,Mapping)
#     ReversibleCounts = MSMLib.EstimateReversibleCountMatrix(CountsAfterTrimming,Prior=Prior)
#     TC = MSMLib.EstimateTransitionMatrix(ReversibleCounts)
#     Populations = np.array(ReversibleCounts.sum(0)).flatten()
#     Populations /= Populations.sum()
#     return (CountsAfterTrimming, ReversibleCounts, TC, Populations,Mapping)

def run(LagTime, assignments, Symmetrize='MLE', Prior=0.0, OutDir="./Data/"):


    FnTProb = os.path.join(OutDir, "tProb.mtx")
    FnTCounts = os.path.join(OutDir, "tCounts.mtx")
    FnTUnSym = os.path.join(OutDir, "tCounts.UnSym.mtx")
    FnMap = os.path.join(OutDir, "Mapping.dat")
    FnAss = os.path.join(OutDir, "Assignments.Fixed.h5")
    FnPops = os.path.join(OutDir, "Populations.dat")
    outputlist = [FnTProb, FnTCounts, FnTUnSym, FnMap, FnAss, FnPops]
    arglib.die_if_path_exists(outputlist)

    n_states = max(assignments.flatten()) + 1
    
    counts_after_trim, rev_counts, t_matrix, populations, mapping = MSMLib.build_msm(assignments,
        lag_time=LagTime, n_states=n_states, symmetrize=Symmetrize,
        sliding_window=True, trimming=True)


    MSMLib.apply_mapping_to_assignments(assignments, mapping)
    n_after_trim = len(np.where( assignments.flatten() != -1 )[0] )
    
    # Print a statement showing how much data was discarded in trimming
    percent = (1.0 - float(n_after_trim) / float(n_states)) * 100.0
    print "WARNING: Ergodic trimming discarded: %f percent of your data" % percent 
 
    # Save all output
    np.savetxt(FnPops, populations)
    np.savetxt(FnMap, mapping,"%d")
    scipy.io.mmwrite(str(FnTProb), t_matrix)
    scipy.io.mmwrite(str(FnTCounts), rev_counts)
    scipy.io.mmwrite(str(FnTUnSym), counts_after_trim)
    Serializer.SaveData(FnAss, assignments)

    for output in outputlist:
        print "Wrote: %s"%output

    return

if __name__ == "__main__":
    parser = arglib.ArgumentParser(description=
"""Estimates the counts and transition matrices from an
Assignments.h5 file. Reversible models can be calculated either from naive
symmetrization or estimation of the most likely reversible matrices (MLE,
recommended). Also calculates the equilibrium populations for the model
produced. Outputs will be saved in the directory of your input Assignments.h5
file.
\nOutput: tCounts.mtx, tProb.mtx, Populations.dat,  Mapping.dat,
Assignments.Fixed.h5, tCounts.UnSym.mtx""")
    parser.add_argument('assignments')
    parser.add_argument('symmetrize', description="""Method by which to estimate a
        symmetric counts matrix. Symmetrization ensures reversibility, but may skew
        dynamics. We recommend maximum likelihood estimation (MLE) when tractable,
        else try Transpose. It is strongly recommended you read the documentation
        surrounding this choice.""", default='MLE',
        choices=['MLE', 'Transpose', 'None'])
    parser.add_argument('lagtime', description='''Lag time to use in model (in
        number of snapshots. EG, if you have snapshots every 200ps, and set the
        lagtime=50, you'll get a model with a lagtime of 10ns)''', type=int)
    parser.add_argument('prior', description='''Strength of Symmetric Prior.
        This prior mitigates the effect of sinks when estimating a reversible
        counts matrix (MLE Estimator).''', default=0.0, type=float)
    parser.add_argument('output_dir')
    args = parser.parse_args()
    
    
    run(args.lagtime, args.assignments['Data'], args.symmetrize, args.prior,
        args.output_dir)
