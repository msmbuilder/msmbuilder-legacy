#!/usr/bin/env python

import os, sys, string, glob, random
import numpy as np

sys.path.append('../')


VERBOSE = False
USE_MATPLOTLIB = True

try:
    import matplotlib
    from pylab import *
except:
    USE_MATPLOTLIB = False


from msmbuilder import Trajectory, Conformation

import DihedralTools
import ASA



if __name__ == '__main__':

    # Parse command-line arguments

    # an examplar PDB needed at the very least to load in atom names, etc, for a Trajectory
    PDBfn = '3GB1.pdb'

    #PDBDir = '/Users/vincentvoelz/Documents/pande/projects/ACBP-unf/cmap_secstruct/pdbfiles_5754_native'
    #HDF5fn = '/Users/vince/projects/WW-reweight/WW-Domain-MSM-Lane2011/data/Gens.lh5'

    Conf=Conformation.Conformation.LoadFromPDB(PDBfn)
    
    print "Dihedral angles (degrees) for the first 10 residues:"
    for i in range(10):
        PhiPsi = DihedralTools.GetSingleResRamaSnap(Conf,ResIndex=i)
        print 'residue', i, 'PhiPsi', PhiPsi
    print 

    print 'Adding a Conf["Radius"] (in nm) to the Conf dictionary:'
    Conf = ASA.AddRadiiToConf(Conf)
    print 'AtomID\tAtomName\tRadius(nm)'
    for i in range(10):
        print '%d\t%s\t%3.3f'%(Conf["AtomID"][i], Conf["AtomNames"][i], Conf["Radius"][i])
    print '...and so on.'
    print 
 
    print 'Calculating accessible surface areas (ASA) for all atoms in 3GB1:' 
    print 'num sphere pts\tTotalArea(nm^2)'
    for n in [5,10,20,50]: 
        areas = ASA.CalculateASA(Conf,n_sphere_point=n)
        print '%d\t%4.4f'%(n, sum(areas))
    print

    print 'Bosco Ho recommends 940 points which can be very time-consuming!'
    print 'Let\'s just calculate the ASA for ResidueID = 12'
    I = np.where( Conf["ResidueID"] == 12 )[0]
    print 'I', I
    print 'num sphere pts\tASA_res12(nm^2)'
    for n in [100,500,1000, 5000, 10000]:
        areas = ASA.CalculateASA(Conf,n_sphere_point=n, ASAIndices=I)
        print '%d\t%4.4f'%(n, sum(areas))
    print

    
    print 'I', I

    

