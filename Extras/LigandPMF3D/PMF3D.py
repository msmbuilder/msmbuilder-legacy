import mdtraj as md
import sys
import pickle
import multiprocessing
import optparse
import pylab
from numpy import *
import glob
import os

# Helper Functions
def eval_distance(mapped_state_distances, cutoff):
    # assumes mapped states and state_distances
    mapped_cutoff_states=[]
    for state in xrange(len(mapped_state_distances)):
        # if mindist to protein > cutoff, add to unbound frames 
        if mapped_state_distances[state] > cutoff:
            mapped_cutoff_states.append(state)
    mapped_cutoff_states=array([int(i) for i in mapped_cutoff_states])
    return mapped_cutoff_states

def get_ligand_minmax(ligcoors, map):
    # build grid of min/max ligand coords from Gens.vmd_ligcoords.dat
    mapped_ligcoors=dict()
    n=0
    for i in xrange(ligcoors.shape[0]):
        if map[i]!=-1:
            mapped_ligcoors[n]=[k*10 for k in ligcoors[i]]
            n+=1
    xmin=100000
    xmax=0
    ymin=100000
    ymax=0
    zmin=100000
    zmax=0
    dx=1
    dy=1
    dz=1
    coordinates=['x', 'y', 'z']
    mins=[xmin, ymin, zmin]
    maxes=[xmax, ymax, zmax]
    for i in sorted(mapped_ligcoors.keys()):
        for j in range(0, len(mapped_ligcoors[i])):
            for (n, k) in enumerate(coordinates):
                if mapped_ligcoors[i][j][n]<=mins[n]:
                    mins[n]=mapped_ligcoors[i][j][n]
                elif mapped_ligcoors[i][j][n]>=maxes[n]:
                    maxes[n]=mapped_ligcoors[i][j][n]
    lengths=dict()
    for (n, i) in enumerate(coordinates):
        lengths[n]=int(((round(maxes[n])+1)-round(mins[n]))/1.0)
    box_volume=lengths[0]*dx*lengths[1]*lengths[2]*dz
    print "box volume sampled by ligand is %s angstroms^3" % box_volume
    total=max(lengths.values())
    ranges=dict()
    for (n, i) in enumerate(coordinates):
        pad=-(lengths[n]-total)
        ranges[n]=range(int(round(mins[n])), int(round(maxes[n]+1+(pad*1.0+1))), 1)
    return mapped_ligcoors, ranges[0], ranges[1], ranges[2], box_volume


def estimate_com_free_energy(modeldir, space, spacetrack, mapped_com_distances, mapped_ligcoors, correction):
    frees=[]
    volumes=[]
    corrs=[]
    axis=[]
    #cutoffs for Protein-Ligand COM distances
    mapped_states=range(0, len(mapped_com_distances))
    cutoffs=arange(0, 20, 1)
    if os.path.exists('%s/frees.dat' % (modeldir)):
        print "free energies already computed for %s" % modeldir
        sys.exit()
    for cutoff in cutoffs:
        bound_frames=where(mapped_com_distances < cutoff)[0]
        if len(bound_frames)==0:
            print "no bound states less than reference com distance %s" % cutoff
            continue
        print "on COM cutoff %s" % cutoff
        new_points={ key: mapped_ligcoors[key] for key in bound_frames}
        new_pops={ key: space.pops[key] for key in bound_frames}
        GD=space.new_manual_allcoor_grid(spacetrack, new_points, new_pops, type='pops')
        boundspace=GD.sum()

        new_pops={ key: 1.0 for key in bound_frames}
        GD=space.new_manual_allcoor_grid(spacetrack, new_points, new_pops, type='pops')
        boundvolume=GD.sum()
        volumes.append(boundspace*boundvolume)
        print "bound volume ", boundspace*boundvolume

        # unbound frames are above COM cutoff
        unbound_frames=array([int(x) for x in mapped_states if x not in bound_frames])
        new_points={ key: mapped_ligcoors[key] for key in unbound_frames}
        new_pops={ key: space.pops[key] for key in unbound_frames}
        GD=space.new_manual_allcoor_grid(spacetrack, new_points, new_pops, type='pops')
        unboundspace=GD.sum()

        # for counting states
        new_pops={ key: 1.0 for key in unbound_frames}
        GD=space.new_manual_allcoor_grid(spacetrack, new_points, new_pops, type='pops')
        unboundvolume=GD.sum()

        # free energy from ratio
        depth=-0.6*log((boundspace*boundvolume)/(unboundspace*boundvolume))
        frees.append(depth)
        axis.append(cutoff)
        print "corrected free energy at COM cutoff %s is %s" % (cutoff, depth+correction)
    k=len(space.pops)
    savetxt('%s/frees.dat' % (modeldir), [x+correction for x in frees])
    savetxt('%s/boundvolumes.dat' % (modeldir), volumes)
    savetxt('%s/axis.dat' % (modeldir), axis)

def get_displacement(coor1, coor2):
    delta = subtract(coor1, coor2)
    return (delta ** 2.).sum(-1) ** 0.5

def get_com_distances(proteinfile, ligandfile, genfile, topology):
    lig_indices=loadtxt(ligandfile, ndmin=1, dtype=int)
    prot_indices=loadtxt(proteinfile, ndmin=1, dtype=int)
    prot_gens=md.load(genfile, top=topology, atom_indices=prot_indices)
    lig_gens=md.load(genfile, top=topology, atom_indices=lig_indices)
    prot_coms=md.compute_center_of_mass(prot_gens)
    lig_coms=md.compute_center_of_mass(lig_gens)
    com_distances=[10*get_displacement(i,j) for (i,j) in zip(prot_coms, lig_coms)]
    return array(com_distances), lig_gens


def convert(GDfree, max):
    frames=where(GDfree==inf)
    for (i,j,k) in zip(frames[0], frames[1], frames[2]):
        GDfree[i,j,k]=max
    return GDfree

# Class for 3D PMF
class PMF3D:
    def __init__(self, pops=None, xaxis=None, yaxis=None, zaxis=None, grid=None):
        self.pops=pops
        self.dx =abs(xaxis[0]-xaxis[1])
        self.dy =abs(yaxis[0]-yaxis[1])
        if zaxis!=None:
            self.dz =abs(zaxis[0]-zaxis[1])
        self.xaxis = array(xaxis)
        self.yaxis = array(yaxis)
        if zaxis!=None:
            self.zaxis = array(zaxis)
        if self.dx==3:
            self.tol=1.5
        elif self.dx==2:
            self.tol=1.0
        elif self.dx==1:
            self.tol=0.5
 
    def microstate_count_grid(self, allcoor):
        spacetrack=dict()
        for i in sorted(allcoor.keys()):
            if i not in spacetrack.keys():
                spacetrack[i]=[]
            for j in range(0, len(allcoor[i])):
                x_location=where((self.xaxis>=round(allcoor[i][j][0]))&(self.xaxis<(round(allcoor[i][j][0])+self.dx)))[0]
                y_location=where((self.yaxis>=round(allcoor[i][j][1]))&(self.yaxis<(round(allcoor[i][j][1])+self.dy)))[0]
                z_location=where((self.zaxis>=round(allcoor[i][j][2]))&(self.zaxis<(round(allcoor[i][j][2])+self.dz)))[0]
                if len(x_location)==0 or len(y_location)==0 or len(z_location)==0:
                    print "no bin available for allcoor[i][j]"
                    import pdb
                    pdb.set_trace()
                if [x_location, y_location, z_location] not in spacetrack[i]:
                    spacetrack[i].append([x_location[0], y_location[0], z_location[0]])
        return spacetrack

    def new_manual_allcoor_grid(self, spacetrack, allcoor, values, type='pops'):
        if type=='pops':
            newgrid=zeros((len(self.xaxis), len(self.yaxis), len(self.zaxis)))
        elif type=='free':
            # pad with max free energy value (pop=0)
            newgrid=max(values)*ones((len(self.xaxis), len(self.yaxis), len(self.zaxis)))
        for (n, i) in enumerate(sorted(allcoor.keys())):
            cn=len(spacetrack[i])
            for j in range(0, len(allcoor[i])):
                x_location=where((self.xaxis>=round(allcoor[i][j][0]))&(self.xaxis<(round(allcoor[i][j][0])+self.dx)))[0]
                y_location=where((self.yaxis>=round(allcoor[i][j][1]))&(self.yaxis<(round(allcoor[i][j][1])+self.dy)))[0]
                z_location=where((self.zaxis>=round(allcoor[i][j][2]))&(self.zaxis<(round(allcoor[i][j][2])+self.dz)))[0]
                if len(x_location)==0 or len(y_location)==0 or len(z_location)==0:
                    print "no bin available for allcoor[i][j]"
                    import pdb
                    pdb.set_trace()
                newgrid[x_location, y_location, z_location]+=(1.0/cn)*values[i]
        return newgrid

 

    def write_dx(self, GD, dir):
        gridspace=1
        newfile=open('%s/pmf3d_d%s.dx' % (dir, gridspace), 'w')
        newfile.write('# Data calculated MSM State Probabilities\n')
        newfile.write('object 1 class gridpositions counts %s %s %s\n' % (GD.shape[0], GD.shape[1], GD.shape[2]))
        newfile.write('origin %s %s %s\n' % (self.xaxis[0], self.yaxis[0], self.zaxis[0]))
        newfile.write('delta %s 0 0\n' % self.dx)
        newfile.write('delta 0 %s 0\n' % self.dy)
        newfile.write('delta 0 0 %s\n' % self.dz)
        newfile.write('object 2 class gridconnections counts %s %s %s\n' % (GD.shape[0], GD.shape[1], GD.shape[2]))
        newfile.write('object 3 class array type double rank 0 items %s data follows\n' % (GD.shape[0]*GD.shape[1]*GD.shape[2]))
        intergrid=zeros((GD.shape[0], GD.shape[1], GD.shape[2]))
        count=0
        for i in range(0, GD.shape[0]):
            for j in range(0, GD.shape[1]):
                for k in range(0, GD.shape[2]):
                    if count==2:
                        if GD[i][j][k]==0:
                            newfile.write('%s\n' % int(GD[i][j][k]))
                        else:
                            newfile.write('%s\n' % GD[i][j][k])
                        count=0
                    else:
                        if GD[i][j][k]==0:
                            newfile.write('%s\t' % int(GD[i][j][k]))
                        else:
                            newfile.write('%s\t' % GD[i][j][k])
                        count+=1
        newfile.write('\nobject "ligand free energy" class field')
        newfile.close()

    def pmfvolume(self, GD):
        sum=0
        oldx=self.xaxis[0]
        oldval=0
        for i in range(0, len(self.xaxis)):
            oldy=self.yaxis[0]
            for j in range(0, len(self.yaxis)):
                oldz=self.zaxis[0]
                for k in range(0, len(self.zaxis)):
                    if abs(self.zaxis[k]-oldz) !=0:
                        if GD[i,j, k] != 0:
                            sum+=GD[i,j, k]*abs(self.xaxis[i]-oldx)*abs(self.yaxis[j]-oldy)*abs(self.zaxis[k]-oldz)
                    oldz=self.zaxis[k]
                oldy=self.yaxis[j]
            oldx=self.xaxis[i]
        return float(sum)
 
