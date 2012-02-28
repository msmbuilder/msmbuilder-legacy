import numpy as np
import scipy
import scipy.sparse

import string

from msmbuilder import MSMLib

def GenerateSpherePoints(n):
    """Returns list of n 3d coordinates of points on a sphere using the
    Golden Section Spiral algorithm."""
    
    inc = np.pi * (3 - np.sqrt(5))
    offset = 2 / float(n)
    k = np.arange(int(n)).reshape(int(n),1)
    y = (k * offset - 1 + (offset / 2)).reshape(int(n),1)
    r = (np.sqrt(1 - y*y)).reshape(int(n),1)
    phi = k * inc
    points = np.concatenate( (np.cos(phi)*r, y, np.sin(phi)*r), axis=1)
    return points

def AddRadiiToConf(Conf):
    """Creates a new dictionary key ["Radius"] (in nm)."""
    
    AtomicRadiiInAngstroms = {   'H': 1.20, 
                'N': 1.55,
                'NA': 2.27,
                'CU': 1.40,
                'CL': 1.75,
                'C': 1.70,
                'O': 1.52,
                'I': 1.98,
                'P': 1.80,
                'B': 1.85,
                'BR': 1.85,
                'S': 1.80,
                'SE': 1.90,
                'F': 1.47,
                'FE': 1.80,
                'K':  2.75,
                'MN': 1.73,
                'MG': 1.73,
                'ZN': 1.39,
                'HG': 1.8,
                'XE': 1.8,
                'AU': 1.8,
                'LI': 1.8,
                '.': 1.8 
               }
    
    Conf["Radius"] = []
    for name in Conf["AtomNames"]:
	atomname = string.upper(name.lstrip('0123456789 '))
	if len(atomname) > 1:
	    if atomname[0:2] == 'NA':
		Conf["Radius"].append(AtomicRadiiInAngstroms['NA']/10.) # convert to nm
	    elif atomname[0:2] == 'CL':
		Conf["Radius"].append(AtomicRadiiInAngstroms['CL']/10.) # convert to nm
	    else:
		Conf["Radius"].append(AtomicRadiiInAngstroms[atomname[0]]/10.) # convert to nm
	else:
	    Conf["Radius"].append(AtomicRadiiInAngstroms[atomname[0]]/10.) # convert to nm
    return Conf


def FindNeighborIndices(Conf, k, ProbeDistance=0.14, NeighborIndices=None):
    """
    Returns list of indices of atoms within probe distance (defaulkt 0.14 nm) to atom k.
    NOTE: NeighborIndices slicing is provided here as a courtesy, so you can ignore solvent
    atoms, for example.  You should include *all* the protein atoms, though, to get an accurate ASA.
    """
    

    print "In FindNeighborIndices().. k =", k
    
    neighbor_indices = []
    radius = Conf["Radius"][k] + 2.*ProbeDistance
    if NeighborIndices == None:
	Indices = range(k)
	Indices.extend(range(k+1, len(Conf["XYZ"])))
    else:
	if not type(NeighborIndices) == type([]):
	    Indices = NeighborIndices.tolist()
	else:
	    Indices = NeighborIndices
	if Indices.count(k) > 0:
	    Indices.remove(k)
    Distances = GetDistancesFromPoint(Conf["XYZ"][k], Conf["XYZ"][Indices])
    return np.array(Indices)[ np.where( Distances < (radius + np.array(Conf["Radius"])[Indices]) ) ]



def GetDistancesFromPoint(p, Pos):
    """Return a list of distances from the xyz point p to each position in Pos."""
    return np.sqrt( GetSqDistancesFromPoint(p, Pos) )

def GetSqDistancesFromPoint(p, Pos):
    """Return a list of squared distances from the xyz point p to each position in Pos."""
    diffs = Pos - p
    return np.array( [np.dot(diffs[i,:], diffs[i,:]) for i in range(diffs.shape[0])] )
    # return np.diag( np.dot(diffs, diffs.T) )

def SqDistance(p1, p2):
    """Return the squared distance between p1 and p2."""
    diff = p1-p2
    return np.dot(diff, diff)

def Distance(p1,p2):
    return np.sqrt( SqDistance(p1, p2) )

def SqDistanceMat(coords):
    """Given an N x 3  numpy.array, return the squared-distance matrix of all coords"""

    o = coords.mean(axis=0)
    centered = coords - o
    crossdot = np.inner(centered, centered)
    selfsq = (centered**2).sum(axis=1)
    irepeat = selfsq.repeat( repeats=crossdot.shape[1] ).reshape( crossdot.shape )
    sqdists = irepeat + irepeat.transpose() - 2.0*crossdot
    return sqdists

def DistanceMat(coords):
    """Given an N x 3  numpy.array, return the distance matrix of all coords"""
    return (SqDistanceMat(coords))**0.5

    
def CalculateASA(Conf, ProbeDistance=0.14, n_sphere_point=960, ASAIndices=None, Verbose=False):
    """
    Returns list of accessible surface areas of the atoms, using the probe
    and atom radius to define the surface.
    """
    
    # Get the radii for each atom, if we haven't yet
    if not Conf.has_key("Radius"):
	Conf = AddRadiiToConf(Conf)
    
    sphere_points = GenerateSpherePoints(n_sphere_point)

    const = 4.0 * np.pi / len(sphere_points)
    areas = []
    if ASAIndices==None:
	ASAIndices=range(len(Conf["XYZ"]))
    else:
        if not type(ASAIndices) == type([]):
            ASAIndices = ASAIndices.tolist()

    AllRadii = np.array(Conf["Radius"])

    FullDistMat = True 
    if FullDistMat:
        Distances = DistanceMat(Conf["XYZ"])

    CalcWithBrokenLoops = True 

    ##########################
    if CalcWithBrokenLoops:

        for i in ASAIndices:

            if FullDistMat:
                AllPossibleNeighbors = ASAIndices[0:i]
                AllPossibleNeighbors.extend( ASAIndices[i+1:] )
                neighbor_indices = np.array(AllPossibleNeighbors)[ np.where( Distances[AllPossibleNeighbors,i] < (Conf["Radius"][i] + 2.*ProbeDistance + AllRadii[AllPossibleNeighbors]) )[0] ]
            else:
                neighbor_indices = FindNeighborIndices(Conf, i, ProbeDistance=ProbeDistance)


            radius = ProbeDistance + Conf["Radius"][i]
            AtomCenteredSpherePoints = Conf["XYZ"][i]+sphere_points*radius

            n_accessible_point = 0
            j_closest_neighbor = 0

            for l in range(AtomCenteredSpherePoints.shape[0]):
                test_point = AtomCenteredSpherePoints[l]
                is_accessible = True

                cycled_indices = range(j_closest_neighbor, len(neighbor_indices))
                cycled_indices.extend(range(j_closest_neighbor))

                for j in cycled_indices:
                    r = Conf["Radius"][neighbor_indices[j]] + ProbeDistance
                    diff_sq = SqDistance(Conf["XYZ"][neighbor_indices[j]], test_point)
                    if diff_sq < r*r:
                        j_closest_neighbor = j
                        is_accessible = False
                        break
                if is_accessible:
                    n_accessible_point += 1

            area = const*n_accessible_point*radius*radius
            if Verbose:
                print 'ASAIndex', i, 'area', area, 'nm^2'
            areas.append(area)

        return areas

    #########################################
    # Calculate with minimum distance count
    else: 
	
        for i in ASAIndices:

            if FullDistMat:
                AllPossibleNeighbors = ASAIndices[0:i]
                AllPossibleNeighbors.extend( ASAIndices[i+1:] )
                neighbor_indices = np.array(AllPossibleNeighbors)[ np.where( Distances[AllPossibleNeighbors,i] < (Conf["Radius"][i] + 2.*ProbeDistance + AllRadii[AllPossibleNeighbors]) )[0] ]
            else:
                neighbor_indices = FindNeighborIndices(Conf, i, ProbeDistance=ProbeDistance)

            n_neighbor = len(neighbor_indices)
            radius = ProbeDistance + Conf["Radius"][i]

	    AtomCenteredSpherePoints = Conf["XYZ"][i]+sphere_points*radius

            accessible = np.ones(len(AtomCenteredSpherePoints))
            for j in range(len(neighbor_indices)):
                r = AllRadii[neighbor_indices[j]] + ProbeDistance
                closest_sq = GetSqDistancesFromPoint(Conf["XYZ"][neighbor_indices[j]], AtomCenteredSpherePoints)
                accessible *= (closest_sq > r*r).astype(int)
            print 'atom', i, 'num neighbors', j, 'num accessible', accessible.sum()
            n_accessible_point = accessible.sum()

            area = const*n_accessible_point*radius*radius 
            if Verbose:
                print 'ASAIndex', i, 'area', area, 'nm^2'
            areas.append(area)

        return areas

