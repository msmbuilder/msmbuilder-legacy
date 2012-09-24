#!/usr/bin/env python
# This file is part of BACE.
#
# Copyright 2012 University of California, Berkeley
#
# BACE is free software; you can redistribute it and/or modify
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
#

version = 1.0

LicenseString = """--------------------------------------------------------------------------------

BACE version %s

Written by Gregory R. Bowman, UC Berkeley

--------------------------------------------------------------------------------

Copyright 2012 University of California, Berkeley.

BACE comes with ABSOLUTELY NO WARRANTY.

BACE is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

--------------------------------------------------------------------------------

Please cite:
GR Bowman. Improved coarse-graining of Markov state models via explicit consideration of statistical uncertainty. J Chem Phys 2012.

Currently available as arXiv:1201.3867 2012.

--------------------------------------------------------------------------------
""" % version

import argparse
import functools
import multiprocessing
import numpy as np
import os
import scipy.io
import scipy.sparse
import logging
logging.basicConfig(format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def getInds(c, stateInds, chunkSize, isSparse, updateSingleState=None):
    indices = []
    for s in stateInds:
        if isSparse:
            dest = np.where(c[s,:].toarray()[0]>1)[0]
        else:
            dest = np.where(c[s,:]>1)[0]
        if updateSingleState != None:
            dest = dest[np.where(dest!=updateSingleState)[0]]
        else:
            dest = dest[np.where(dest>s)[0]]
        if dest.shape[0] == 0:
            continue
        elif dest.shape[0] < chunkSize:
            indices.append((s, dest))
        else:
            i = 0
            while dest.shape[0] > i:
                if i+chunkSize > dest.shape[0]:
                    indices.append((s, dest[i:]))
                else:
                    indices.append((s, dest[i:i+chunkSize]))
                i += chunkSize
    return indices

def run(c, nMacro, nProc, multiDist, outDir, filterFunc, chunkSize=100):
    # perform filter
    logger.info("Checking for states with insufficient statistics")
    c, map, statesKeep = filterFunc(c, nProc)

    # get num counts in each state (or weight)
    w = np.array(c.sum(axis=1)).flatten()
    w[statesKeep] += 1

    unmerged = np.zeros(w.shape[0], dtype=np.int8)
    unmerged[statesKeep] = 1

    # get nonzero indices in upper triangle
    indRecalc = getInds(c, statesKeep, chunkSize, scipy.sparse.issparse(c))
    if scipy.sparse.issparse(c):
        dMat = scipy.sparse.lil_matrix(c.shape)
    else:
        dMat = np.zeros(c.shape, dtype=np.float32)

    if scipy.sparse.issparse(c):
        c = c.tocsr()

    i = 0
    nCurrentStates = statesKeep.shape[0]
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    fBayesFact = open("%s/bayesFactors.dat" % outDir, 'w')
    dMat, minX, minY = calcDMat(c, w, fBayesFact, indRecalc, dMat, nProc, statesKeep, multiDist, unmerged, chunkSize)
    logger.info("Coarse-graining...")
    while nCurrentStates > nMacro:
        logger.info("Iteration %d, merging %d states", i, nCurrentStates)
        c, w, indRecalc, dMat, map, statesKeep, unmerged, minX, minY = mergeTwoClosestStates(c, w, fBayesFact, indRecalc, dMat, nProc, map, statesKeep, minX, minY, multiDist, unmerged, chunkSize)
        nCurrentStates -= 1
        np.savetxt("%s/map%d.dat" % (outDir, nCurrentStates), map, fmt="%d")
        i += 1
    fBayesFact.close()

def mergeTwoClosestStates(c, w, fBayesFact, indRecalc, dMat, nProc, map, statesKeep, minX, minY, multiDist, unmerged, chunkSize):
    cIsSparse = scipy.sparse.issparse(c)
    if cIsSparse:
        c = c.tolil()
    if unmerged[minX]:
        c[minX,statesKeep] += unmerged[statesKeep]*1.0/c.shape[0]
        unmerged[minX] = 0
        if cIsSparse:
            c[statesKeep,minX] += np.matrix(unmerged[statesKeep]).transpose()*1.0/c.shape[0]
        else:
            c[statesKeep,minX] += unmerged[statesKeep]*1.0/c.shape[0]
    if unmerged[minY]:
        c[minY,statesKeep] += unmerged[statesKeep]*1.0/c.shape[0]
        unmerged[minY] = 0
        if cIsSparse:
            c[statesKeep,minY] += np.matrix(unmerged[statesKeep]).transpose()*1.0/c.shape[0]
        else:
            c[statesKeep,minY] += unmerged[statesKeep]*1.0/c.shape[0]
    c[minX,statesKeep] += c[minY,statesKeep]
    c[statesKeep,minX] += c[statesKeep,minY]
    c[minY,statesKeep] = 0
    c[statesKeep,minY] = 0
    dMat[minX,:] = 0
    dMat[:,minX] = 0
    dMat[minY,:] = 0
    dMat[:,minY] = 0
    if cIsSparse:
        c = c.tocsr()
    w[minX] += w[minY]
    w[minY] = 0
    statesKeep = statesKeep[np.where(statesKeep!=minY)[0]]
    indChange = np.where(map==map[minY])[0]
    map = renumberMap(map, map[minY])
    map[indChange] = map[minX]
    indRecalc = getInds(c, [minX], chunkSize, cIsSparse, updateSingleState=minX)
    dMat, minX, minY = calcDMat(c, w, fBayesFact, indRecalc, dMat, nProc, statesKeep, multiDist, unmerged, chunkSize)
    return c, w, indRecalc, dMat, map, statesKeep, unmerged, minX, minY

def renumberMap(map, stateDrop):
    for i in xrange(map.shape[0]):
         if map[i] >= stateDrop:
             map[i] -= 1
    return map

def calcDMat(c, w, fBayesFact, indRecalc, dMat, nProc, statesKeep, multiDist, unmerged, chunkSize):
    nRecalc = len(indRecalc)
    if nRecalc > 1 and nProc > 1:
        if nRecalc < nProc:
            nProc = nRecalc
        pool = multiprocessing.Pool(processes=nProc)
        n = len(indRecalc)
        stepSize = int(n/nProc)
        if n % stepSize > 3:
            dlims = zip(range(0,n,stepSize), range(stepSize,n,stepSize)+[n])
        else:
            dlims = zip(range(0,n-stepSize,stepSize), range(stepSize,n-stepSize,stepSize)+[n])
        args = []
        for start,stop in dlims:
            args.append(indRecalc[start:stop])
        result = pool.map_async(functools.partial(multiDist, c=c, w=w, statesKeep=statesKeep, unmerged=unmerged, chunkSize=chunkSize), args)
        result.wait()
        d = np.vstack(result.get())
        pool.close()
    else:
        d = multiDist(indRecalc, c, w, statesKeep, unmerged, chunkSize)
    for i in xrange(len(indRecalc)):
        dMat[indRecalc[i][0], indRecalc[i][1]] = d[i][:len(indRecalc[i][1])]

    # BACE BF inverted so can use sparse matrices
    if scipy.sparse.issparse(dMat):
        minX = minY = -1
        maxD = 0
        for x in statesKeep:
            if len(dMat.data[x]) == 0:
                continue
            pos = np.argmax(dMat.data[x])
            if dMat.data[x][pos] > maxD:
                maxD = dMat.data[x][pos]
                minX = x
                minY = dMat.rows[x][pos]
    else:
        indMin = dMat.argmax()
        minX = np.floor(indMin / dMat.shape[1])
        minY = indMin % dMat.shape[1]

    fBayesFact.write("%d %f\n" % (statesKeep.shape[0]-1, 1./dMat[minX,minY]))
    return dMat, minX, minY

def multiDistDense(indicesList, c, w, statesKeep, unmerged, chunkSize):
    d = np.zeros((len(indicesList),chunkSize), dtype=np.float32)
    for j in xrange(len(indicesList)):
        indices = indicesList[j]
        ind1 = indices[0]
        c1 = c[ind1,statesKeep] + unmerged[ind1]*unmerged[statesKeep]*1.0/c.shape[0]
        d[j,:indices[1].shape[0]] = 1./multiDistDenseHelper(indices[1], c1, w[ind1], c, w, statesKeep, unmerged) # BACE BF inverted so can use sparse matrices
    return d

def multiDistDenseHelper(indices, c1, w1, c, w, statesKeep, unmerged):
    d = np.zeros(indices.shape[0], dtype=np.float32)
    p1 = c1 / w1
    for i in xrange(indices.shape[0]):
        ind2 = indices[i]
        c2 = c[ind2,statesKeep] + unmerged[ind2]*unmerged[statesKeep]*1.0/c.shape[0]
        p2 = c2 / w[ind2]
        cp = c1 + c2
        cp /= (w1 + w[ind2])
        d[i] = c1.dot(np.log(p1/cp)) + c2.dot(np.log(p2/cp))
    return d

def multiDistSparse(indicesList, c, w, statesKeep, unmerged, chunkSize):
    d = np.zeros((len(indicesList),chunkSize), dtype=np.float32)
    for j in xrange(len(indicesList)):
        indices = indicesList[j]
        ind1 = indices[0]
        c1 = c[ind1,statesKeep].toarray()[0] + unmerged[ind1]*unmerged[statesKeep]*1.0/c.shape[0]
        d[j,:indices[1].shape[0]] = 1./multiDistSparseHelper(indices[1], c1, w[ind1], c, w, statesKeep, unmerged) # BACE BF inverted so can use sparse matrices
    return d

def multiDistSparseHelper(indices, c1, w1, c, w, statesKeep, unmerged):
    d = np.zeros(indices.shape[0], dtype=np.float32)
    p1 = c1 / w1
    for i in xrange(indices.shape[0]):
        ind2 = indices[i]
        c2 = c[ind2,statesKeep].toarray()[0] + unmerged[ind2]*unmerged[statesKeep]*1.0/c.shape[0]
        p2 = c2 / w[ind2]
        cp = c1 + c2
        cp /= (w1 + w[ind2])
        d[i] = c1.dot(np.log(p1/cp)) + c2.dot(np.log(p2/cp))
    return d

def add_argument(group, *args, **kwargs):
    if 'default' in kwargs:
        d = 'Default: {d}'.format(d=kwargs['default'])
        if 'help' in kwargs:
            kwargs['help'] += ' {d}'.format(d=d)
        else:
            kwargs['help'] = d 
    group.add_argument(*args, **kwargs)

def filterFuncDense(c, nProc):
    # get num counts in each state (or weight)
    w = np.array(c.sum(axis=1)).flatten()
    w += 1

    # init map from micro to macro states
    map = np.arange(c.shape[0], dtype=np.int32)

    # pseudo-state (just pseudo counts)
    pseud = np.ones(c.shape[0], dtype=np.float32)
    pseud /= c.shape[0]

    indices = np.arange(c.shape[0], dtype=np.int32)
    statesKeep = np.arange(c.shape[0], dtype=np.int32)
    unmerged = np.ones(c.shape[0], dtype=np.float32)

    nInd = len(indices)
    if nInd > 1 and nProc > 1:
        if nInd < nProc:
            nProc = nInd
        pool = multiprocessing.Pool(processes=nProc)
        stepSize = int(nInd/nProc)
        if nInd % stepSize > 3:
            dlims = zip(range(0,nInd,stepSize), range(stepSize,nInd,stepSize)+[nInd])
        else:
            dlims = zip(range(0,nInd-stepSize,stepSize), range(stepSize,nInd-stepSize,stepSize)+[nInd])
        args = []
        for start,stop in dlims:
            args.append(indices[start:stop])
        result = pool.map_async(functools.partial(multiDistDenseHelper, c1=pseud, w1=1, c=c, w=w, statesKeep=statesKeep, unmerged=unmerged), args)
        result.wait()
        d = np.concatenate(result.get())
        pool.close()
    else:
        d = multiDistDenseHelper(indices, pseud, 1, c, w, statesKeep, unmerged)

    # prune states with Bayes factors less than 3:1 ratio (log(3) = 1.1)
    statesPrune = np.where(d<1.1)[0]
    statesKeep = np.where(d>=1.1)[0]
    logger.info("Merging %d states with insufficient statistics into their kinetically-nearest neighbor", statesPrune.shape[0])

    for s in statesPrune:
        dest = c[s,:].argmax()
        c[dest,:] += c[s,:]
        c[s,:] = 0
        c[:,s] = 0
        map = renumberMap(map, map[s])
        map[s] = map[dest]

    return c, map, statesKeep

def filterFuncSparse(c, nProc):
    # get num counts in each state (or weight)
    w = np.array(c.sum(axis=1)).flatten()
    w += 1

    # init map from micro to macro states
    map = np.arange(c.shape[0], dtype=np.int32)

    # pseudo-state (just pseudo counts)
    pseud = np.ones(c.shape[0], dtype=np.float32)
    pseud /= c.shape[0]

    indices = np.arange(c.shape[0], dtype=np.int32)
    statesKeep = np.arange(c.shape[0], dtype=np.int32)
    unmerged = np.ones(c.shape[0], dtype=np.int8)

    nInd = len(indices)
    if nInd > 1 and nProc > 1:
        if nInd < nProc:
            nProc = nInd
        pool = multiprocessing.Pool(processes=nProc)
        stepSize = int(nInd/nProc)
        if nInd % stepSize > 3:
            dlims = zip(range(0,nInd,stepSize), range(stepSize,nInd,stepSize)+[nInd])
        else:
            dlims = zip(range(0,nInd-stepSize,stepSize), range(stepSize,nInd-stepSize,stepSize)+[nInd])
        args = []
        for start,stop in dlims:
            args.append(indices[start:stop])
        result = pool.map_async(functools.partial(multiDistSparseHelper, c1=pseud, w1=1, c=c, w=w, statesKeep=statesKeep, unmerged=unmerged), args)
        result.wait()
        d = np.concatenate(result.get())
        pool.close()
    else:
        d = multiDistSparseHelper(indices, pseud, 1, c, w, statesKeep, unmerged)

    # prune states with Bayes factors less than 3:1 ratio (log(3) = 1.1)
    statesPrune = np.where(d<1.1)[0]
    statesKeep = np.where(d>=1.1)[0]
    logger.info("Merging %d states with insufficient statistics into their kinetically-nearest neighbor", statesPrune.shape[0])

    for s in statesPrune:
        dest = c.rows[s][np.argmax(c.data[s])]
        c[dest,:] += c[s,:]
        c[s,:] = 0
        c[:,s] = 0
        map = renumberMap(map, map[s])
        map[s] = map[dest]

    return c, map, statesKeep

if __name__ == '__main__':
    print LicenseString
    parser = argparse.ArgumentParser(description='''Bayesian agglomerative clustering engine (BACE) for coarse-graining MSMs.  For example, building macrostate models from microstate models.

The algorithm works by iteratively merging states until the final desired number of states (the nMacro parameter) is reached.

Description output (put into directory specified with outDir): 
    bayesFactors.dat = the Bayes factors (cost) for each merging of two states
    mapX.dat = the mapping from the original state numbering to X coarse-grained states'''

, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_argument(parser, '-c', dest='tCountFn', help='Path to transition count matrix file (sparse and dense formats accepted).', required=True)
    add_argument(parser, '-n', dest='nMacro', help='Minimum number of macrostates to make.', default=2, type=int)
    add_argument(parser, '-p', dest='nProc', help='Number of processors to use (if None, the code will use all the available processors on your machine).', default=None, type=int, required=False)
    add_argument(parser, '-f', dest='forceDense', help='If true, will force transition matrix into a dense format. Using the dense format is faster if you have enough memory.', default=False, type=bool, required=False, nargs='?', const=True)
    add_argument(parser, '-o', dest='outDir', help='Path to save output to.', default="Output_BACE", required=False)
    args = parser.parse_args()

    if "mtx" in args.tCountFn:
        c = scipy.sparse.lil_matrix(scipy.io.mmread(args.tCountFn), dtype=np.float32)
        multiDist = multiDistSparse
        filterFunc = filterFuncSparse
        if args.forceDense:
            logger.info("Forcing dense")
            c = c.toarray()
            multiDist = multiDistDense
            filterFunc = filterFuncDense
    else:
        c = np.loadtxt(tCountFn, dtype=np.float32)
        multiDist = multiDistDense
        filterFunc = filterFuncDense

    if args.nProc == None:
        args.nProc = multiprocessing.cpu_count()
    logger.info("Set number of processors to %s", args.nProc)

    run(c, args.nMacro, args.nProc, multiDist, args.outDir, filterFunc, chunkSize=100)


