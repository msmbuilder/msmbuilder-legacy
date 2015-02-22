import mdtraj as md
import PMF3D
import optparse
import pylab
from numpy import * 
import glob
import os
import sys
import pickle
from scipy import interpolate, integrate

"""
LigandPMF3D
===========

** This branch to be used with MSMBuilder 2.8 **

Python program for creating 3D PMFs and estimating binding free energies from protein-ligand binding simulations.

Program requires as input (see options with -h flag):
* the path to the directory with MSM output (Populations.dat, Mapping.dat). This directory will also be used for the PMF and free energy output.
* the generators trajectory file, in any format readble by MDTraj. The computed ligand-protein center of mass (COM) distances will be saved here with the same prefix (i.e. for Gens.lh5, output it Gens.com.dat)
* the index file for ligand atoms.
* the index file for protein atoms, which are used for calculating ligand-protein center of mass (COM) distances and binding free energies (we suggest active site atoms only).
* topology, a reference PDB file.

Program output is a OpenDX file pmf3d_d1.dx (3D PMF at gridspace 1 Angstrom) centered on the protein as aligned in the generators trajectory.
This file can be loaded into VMD:
* you can run the command: vmd -pdb reference.pdb -dx pmf3d_d1.dx
* make a new representation in Isosurface, and scale the isovalues ranging from the PMF minimum at 0, up to the max.
If your 3D PMF is not centered on your protein, check that your reference structure is aligned to your generators file.

An additional -f option will output estimated binding free energies as a function of ligand-protein COM distance. This prints:
* the standard state correction computed from an estimated box size using ligand coordinates: standard_correction.dat
* standard corrected, binding free energyes: frees.dat
* population-weighted bound volumes (which should plateau as you increase your cutoff): boundvolumes.dat 
* the COM distances corresponding to the free energies and volumes: axis.dat 
"""

def main(modeldir, genfile, ligandfile, proteinfile, topology, writefree=False):
    dir=os.path.dirname(genfile)
    filename=genfile.split('%s/' % dir)[1].split('.')[0]
    gens=md.load(genfile, top=topology)

    print "computing 3D PMF from model in %s" % modeldir
    map=loadtxt('%s/Mapping.dat' % modeldir)
    pops=loadtxt('%s/Populations.dat' % modeldir)
    ligandind=loadtxt(ligandfile, dtype=int, ndmin=1)

    print "Tracking ligand sampling"
    # also computes COM distances for free energy estimation, if desired
    com_distances, lig_gens=PMF3D.get_com_distances(proteinfile, ligandfile, genfile, topology)
    mapped_ligcoors, x_range, y_range, z_range, box_volume=PMF3D.get_ligand_minmax(lig_gens.xyz, map)

    # Generate Mapped 3-D PMF
    frames=where(map!=-1)[0]
    mapped_states=map[frames]
    space=PMF3D.PMF3D(pops, x_range, y_range, z_range)
    spacetrack=space.microstate_count_grid(mapped_ligcoors)
    new_pops={ key: pops[n] for (n, key) in enumerate(mapped_ligcoors.keys())}
    GD=space.new_manual_allcoor_grid(spacetrack, mapped_ligcoors, new_pops, type='pops')
    free=array([-0.6*log(i) for i in pops])
    subtract=min(free)
    free=array([k-subtract for k in free])
    GD=space.new_manual_allcoor_grid(spacetrack, mapped_ligcoors, new_pops, type='pops')
    GDfree=-0.6*log(GD)
    GDfree=PMF3D.convert(GDfree, max(free))
    GDfree=GDfree-min(GDfree.flatten())
    space.write_dx(GDfree, modeldir)
    if writefree==True:
        print "writing COM output for %s to %s.com.dat" % (dir, filename)
        savetxt('%s/%s.com.dat' % (dir, filename), com_distances)
        mapped_com_distances=com_distances[frames]
        correction=-0.6*log(box_volume/1600.0)
        print "standard correction %s" % correction
        savetxt('%s/standard_correction.dat' % modeldir, array([correction,]))
        PMF3D.estimate_com_free_energy(modeldir, space, spacetrack,mapped_com_distances, mapped_ligcoors, correction)



def parse_commandline():
    parser = optparse.OptionParser()
    parser.add_option('-m', '--modeldir', dest='modeldir',
                      help='directory with MSM')
    parser.add_option('-g', '--genfile', dest='genfile',
                      help='input gens file lh5')
    parser.add_option('-l', '--ligandfile', dest='ligandfile',
                      help='ligand index file')
    parser.add_option('-p', '--proteinfile', dest='proteinfile',
                      help='protein binding site residue index file')
    parser.add_option('-t', '--topology', dest='topology',
                      help='reference topology')
    parser.add_option('-f', action="store_true", dest="writefree", help='perform free energy calc with P-L COM distances')
    (options, args) = parser.parse_args()
    return (options, args)

#run the function main if namespace is main
if __name__ == "__main__":
    (options, args) = parse_commandline()
    if options.writefree==True:
        main(modeldir=options.modeldir, genfile=options.genfile, ligandfile=options.ligandfile, proteinfile=options.proteinfile, topology=options.topology, writefree=True)
    else:
        main(modeldir=options.modeldir, genfile=options.genfile, ligandfile=options.ligandfile, proteinfile=options.proteinfile, topology=options.topology)

