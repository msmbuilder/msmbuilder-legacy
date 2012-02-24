from numpy import *
import numpy as np
import multiprocessing as mp
from time import time 
from msmbuilder import Trajectory 

def getContactMapHelper( args ):
	"""
	This is a helper function for using multiprocessing and getContactMap
	"""
	XYZ = args[0]
	resList = args[1]
	cutoff = args[2] 

	return getContactMap( XYZ, resList = resList, cutoff = cutoff )

def getContactMap( XYZ, resList = None, cutoff = 0.6 ):
	"""
	This function calculates the contact map for a given XYZ coordinate of atoms. NOTE: This calculates contacts between atoms unless resList is given which specifies to calculate residue contacts by the closest distance between atoms of any two residues.

	Input:
	1) XYZ - np.array of xyz coordinates for the atoms in question ( units = nm )
	2) resList [ None ] - list of np.arrays of indices in XYZ corresponding to which atoms correspond to which residue. So the output matrix will be N x N where N = len( resList ).
	3) cutoff [ 0.6 ] - Cutoff to use in calculating the contacts ( units = nm )

	Output:
	3) ContactMap - np.array of size N x N, which contains only the upper triangular values in the contact map minus the diagonal, and 2 off-diagonals (corresponding to residues 3 places away from eachother)
	"""
	cutoff2 = cutoff * cutoff # Avoid the sqrt, and compare to cutoff**2
	# If no resList, then assume each atom represents a residue
	if resList == None:
		resList = np.array( [ [i] for i in range( len( XYZ ) ) ] )
  
	PW_DistAtoms = np.ones( ( len( XYZ ), len( XYZ ) ) ) * cutoff2 
	Inter = np.zeros( ( len( resList ), len( XYZ ) ) ) 
	PW_Dist = np.zeros( ( len( resList ), len( resList ) ) )  
	# In the end will set to zero anything that is BIGGER than the cutoff, so the diagonals should be set to the cutoff2 so that they are always set to 1
   
	# We will determine the contacts by computing the pairwise distances:
	for ind, atom in enumerate( XYZ[:-len( resList[-1] ) ] ): # Go up to the last residue's atoms
	# First find out which residue this atom is in:
		inRes = np.where( np.array( [ ( ind in res ) for res in resList ] ) )[0]
		if len( inRes ) != 1:
			print "atom %d in multiple residues... Exiting..."
			exit()
		inRes = inRes[0]

		# Since we only need the upper triangular, three diagonals we need only calculate the distance to a portion of the other atoms.
		#print resList[ inRes + 1 : ]
		#print XYZ[ resList[ inRes + 1 : ] ]
		TestInd = np.concatenate( resList[ inRes + 1 : ] ) # These are the indices of the residues greater than this index's residue
		TestAtoms = np.row_stack( XYZ[ TestInd ] ) # These are the coordinates for the aboe residues
		diff = TestAtoms - atom
		dists = ( diff * diff ).sum( axis = 1 )

		PW_DistAtoms[ ind, TestInd ] = dists
		PW_DistAtoms[ TestInd, ind ] = dists
		
	PW_DistAtoms = (PW_DistAtoms <= cutoff2)
	# since the atoms belong to residues, we have to turn each block into a signle value based on whether that block as all zeros or at least one one. If it has a one in it, then the residues are contacting
	for i in range( len( resList ) ):
		Inter[ i, : ] = PW_DistAtoms[ resList[i] , : ].any( axis = 0 ) # First contract axis = 0 ( rows )
		
	for i in range( len( resList ) ):
		PW_Dist[ i, : ] = Inter[ :, resList[i] ].any( axis = 1 ) # Then contract axis = 1 ( columns )
	
	return PW_Dist

def calcContacts( traj, procs = 1, cutoff = 0.6, RestrictToCA=False):
	"""
	This function calculates the contact map for an entire trajectory, and returns it as a np.array.

	Inputs:
	1) traj - Trajectory.Trajectory object
	2) procs [ 1 ] - number of processes to run using python.multiprocessing
	3) cutoff [ 0.6 ] - cutoff to define a contact, NOTE: this cutoff is the distance between any two atoms in a trajectory.
	4) RestrictToCA [ False ] - If true then the trajectory will be restricted to only CA's which means there is only one distance to calculate between residues. This will be faster, but since some residues are different sizes, a constant cutoff doesn't really make sense...

	Outputs:
	1) contactMaps - np.array of dimension ( N, m, m ) where N = number of frames in traj, and m is the number of residues in traj 
	"""

	pool = mp.Pool( procs )

	if RestrictToCA:
		traj.RestrictAtomIndices( np.where( traj['AtomNames'] == 'CA' )[0] )

	# Need to produce a residue list corresponding to the atom indices in each residue for this to work.
	# Doesn't matter how the residues are numbered, as long as they are IN ORDER... this should be a safe assumption...
	resIDs = traj['ResidueID']
	if ( ( resIDs[1:] - resIDs[:-1] ) < 0 ).any():
		print "Residue IDs in trajectory are NOT in order! Fix this and then call this function again. You shouldn't have miss-ordered residue indices anyway..."
		exit()

	uniqueResIDs = np.unique( resIDs ) # This SORTs the id's so if they are not in order, then we are in trouble...
	resList = [ np.where( resIDs == i )[0] for i in uniqueResIDs ] # Get the indices for each residue

	print traj['XYZList'].shape
	N = traj['XYZList'].shape[0]

	#result = pool.map_async( getContactMap, traj['XYZList'] )
	result = pool.map_async( getContactMapHelper, zip( traj['XYZList'], N * [ resList ], N * [ cutoff ] ) )
	pool.close()
	pool.join()
	sol = result.get()

	contactMaps = np.array( sol )

	return contactMaps


