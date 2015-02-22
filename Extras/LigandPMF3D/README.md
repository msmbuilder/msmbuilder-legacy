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

