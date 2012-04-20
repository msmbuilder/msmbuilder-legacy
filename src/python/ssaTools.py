#===============================================================================
# ssaTools.py
#
# Please reference
# N Singhal Hinrichs, and VS Pande. J. Chem. Phys. 2007. Calculation of the 
# distribution of eigenvalues and eigenvectors in Markovian State Models for
# molecular dynamics.
#
# Written 10/1/10 by
# Dan Ensign <densign@mail.utexas.edu>
# Gregory Bowman <gregoryrbowman@gmail.com>
# Sergio Bacallado <sergiobacallado@gmail.com>
# Stanford University
# Pande group
#
# Copyright (C) 2008  Stanford University
#
# This program is free software; you can redistribute it and/or modify
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
#===============================================================================
# TODO:
#===============================================================================
# GLOBAL IMPORTS:
from scipy import identity, matrix, zeros
from scipy.linalg import eigvals, lu
from random import gammavariate, random
from math import fabs
from copy import copy
from os import mkdir
from os.path import exists
#===============================================================================
# LOCAL IMPORTS:
#===============================================================================


def openmat( filename ):
	FILE = open( filename )
	text = FILE.readlines()
	FILE.close()
	elems = []
	for line in text :
		row = []
		vals = line.split()
		for val in vals :
			row.append( int( val ) )

		elems.append( tuple( row ) )

	return tuple( elems )

def show( arr, label = "array", format = "% .6e " ):
        #format = "% 3.8f "
        nindex = arr.shape[0]
        s = "%s =\t" % label
        for i in range( nindex ):
                for j in range( nindex ):
                        elem = arr[i,j]
                        s += format % elem
                s += "\n\t"
        print s

def mathform( arr, format="%10.10f" ):
	s = "{ "
	arr = arr2lst( arr )

	try:
		dim = len( arr )
	except:
		dim = 0 

	if dim == 0 : 
		# just a number
		return format % arr

	elif dim == 1 :
		# either a 1-d array
		for i in range( dim ):
			s += format % arr[ i ]

			if i == dim-1 :
				s += " }"
			else:
				s += ", "
	else:
		for i in range( dim ):
			s += mathform( arr[ i ] ) 
			if i == dim-1 :
				s += " }"
			else:
				s += ", "

        return s

def makemat( dim, r=10, offset=0 ):
        a = zeros( (dim, dim), "float64" )
        for i in range( dim ):
                for j in range( dim ):
                        n = float( int( r*random() - offset ) )
                        a[i][j] = n
        return a

def sumNormalize( u ):
	v = copy( u )
	tot = float( u.sum() )
	return v/tot

def normalizeRows( mat ):
        # divide each row by the sum of the entries.
        dim = mat.shape[0]
        newmat = []
        i = 0
        while i < dim :
                row = mat[ i,0: ]
                newmat.append( arr2lst( sumNormalize( row ) ) )
                i += 1
	newmat = matrix( newmat )
        return newmat

def swapcol( mat, x, y ):
        # this will actually modify mat
        newcolx = copy( mat[ 0:,y ] )
        newcoly = copy( mat[ 0:,x ] )
        mat[0:,x] = newcolx
        mat[0:,y] = newcoly

def swaprow( mat, x, y ):
        # this will actually modify mat 
        newrowx = copy( mat[ y,0: ] )
        newrowy = copy( mat[ x,0: ] )
        mat[ x,0: ] = newrowx
        mat[ y,0: ] = newrowy

def findsmallestdiag( mat ):
        # which is the smallest (absolute) value on the diagonal of mat?
	diag = abs( matrix( mat ).diagonal() )
	minIndex = diag.argmin()

        return minIndex

def arr2lst( arr ):
        try:
		if arr.shape[0] == 1 :
			lst = arr.tolist()[0]
		else:
        		lst = arr.tolist()

        except AttributeError:
                lst = arr
        return lst
                  
def charpolmatrix( M, evalIndex = 0 ):
	# return a matrix A = M - l*I (l is the eigenvalue with index evalIndex)
	dim = M.shape[ 0 ]
	evals = eigvals( M )
	myeval = evals[ evalIndex ]
	return M - identity( dim )*myeval 

def decompose( matrix ):
	# Returns the decomposition of a matrix A where
	#
	# Q.A.Q = P.L.U
	#
	# P.L.U is the factoring of Q.A.Q such that L is a lower triangular matrix with 1's
	# on the diagonal and U is an upper triangular matrix; P is the permutation (row-swapping
	# operations) required for this procedure. The permutation matrix Q is chosen such that 
	# the last element of U is its smallest diagnoal element. If A has a zero eigenvalue, 
	# then U's last element will be zero.
	
	dim = matrix.shape[ 0 ]

	# first decomposition
	( P, L, U ) = lu( matrix )
	
 	# detect the smallest element of U
	smallestIndex = findsmallestdiag( U )
	smallest = U[ smallestIndex, smallestIndex ]

	#show( matrix, "M" )
	#show( U, "U" )
	#print "Smallest element is %f at %d" % ( smallest, smallestIndex )

	# is the permutation Q not just the identity matrix?
	Q = identity( dim )
	if smallestIndex+1 != dim :
		# trick: exchange row 'smallestIndex' with row 'dim-1' of the identity matrix
		swaprow( Q, smallestIndex, dim-1 )

	return ( P, L, U, Q )

class DirichletDistribution( object ):
	# Dirichlet distribution
	# alphas are a list of reals > 0 (counts)
	#
	# methods:
	# 	sample() - return a random sample from the distribution

	def __init__( self, counts ):
		
		self.alphas = matrix(counts) # + 1 # Sergio: commented out the +1 in order to make
						   # the input of ssaCalculator (priorCounts), the alpha 
						   # parameters instead of "pseudo counts".
		
		self.nparams = self.alphas.shape[ 1 ]
		self.alphasSum = self.alphas.sum()

	def sample( self ):
		# sample vector X = (x1, x2, ..., xK )
		# generated from xi = yi/Ytot
		# with each yi sampled from gamma distribution
		# 	p( yi ) = gammavariate( alphai, 1 )
		# and Ytot the sum of the yi's.

		ylist = []
		ysum = 0
		n = 0
		while n < self.nparams :
			alpha = self.alphas[ 0, n ]
			yi = gammavariate( alpha, 1 )
			ylist.append( yi )
			ysum += yi 
			n += 1
		
		xlist = []
		n = 0
		while n < self.nparams :
			xlist.append( ylist[ n ]/ysum )
			n += 1

		return tuple( xlist )

	def mean( self, index ):
		# return the expectation of parameter i: ai/aSum
		return float( self.alphas[ 0,index ] ) / float( self.alphasSum )

	def var( self, index ):
		# return the variance of parameter i:  
		ai = float( self.alphas[ index ] )
		a0 = float( self.alphasSum )
		num = ai*(a0 - ai )
		den = a0**2 * ( a0+1 )
		return num/den

"""
if __name__ == "__main__" :
	import sys
	ntot = int( sys.argv[1] )
	alist = sys.argv[2:]
	a = []
	for elem in alist:
		a.append( int( elem ) )
	a = tuple( a )
	d = DirichletDistribution( a )

	sumvector = []
	i = 0
	while i < len( a ):
		sumvector.append( 0 )
		i += 1

	for i in range( ntot ):
		sample = d.sample()
		
		l = 0
		while l < len( a ) :
			sumvector[ l ] += sample[ l ]
			l += 1

	s = ""
	l = 0
	while l < len( a ):
		rat = sumvector[ l ] / ntot
		s += "%6.6f " % rat
		l += 1
	print s
"""
