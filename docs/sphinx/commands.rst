MSMBuilder Commands
-------------------

The MSMBuilder commands are listed below. Each command corresponds to a
single scrip that can be called either through ``msmb`` or on its own.
Each command also provides instructions by running with the -h flag
(e.g. ``msmb Cluster -h`` or ``Cluster.py -h``). Note that the installer
(setup.py) should have installed each script below to someplace in your
PATH.

msmb
~~~~
All of the individual msmbuilder commands can be accessed as subcommands from this script, i.e. ``msmb ConvertDataToHDF``. Using ``msmb -h``, you can get
a list of all of the available msmbuilder commands.

ConvertDataToHDF.py
~~~~~~~~~~~~~~~~~~~

Merges sequences of XTC or DCD files into HDF5 files that MSMB2 can read
quickly. Takes data from a directory of trajectory directories or a
FAH-style filesystem.

CreateAtomIndices.py
~~~~~~~~~~~~~~~~~~~~

Selects atom indices you care about and dumps them into a flat text
file. Can select all non-symmetric atoms, all heavy atoms, all alpha
carbons, or all atoms.

Cluster.py
~~~~~~~~~~

Cluster your data using your choice of clustering algorithm and distance
metric. We have previously used several clustering protocols, which are
summarized:

#. RMSD + k-centersCluster.py rmsd )

#. RMSD + hybrid k-centers / k-medoidsCluster.py rmsd
   )

#. RMSD + WardCluster.py rmsd )

Note that Ward clustering calculates an :math:`O(N^2)` distance matrix,
which may be prohibitive for datasets with many conformations.

Most of our experience has been in applying MSMBuilder to protein
folding. Thus, non-folding applications may require a slightly different
protocol.

Assign.py / AssignHierarchical.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assign.py assigns data to the cluster generators calculated using the
k-centers or hybrid algorithms.

AssignHierarchical.py assigns data using the output of a hierarchical
clustering algorithm such as Ward. The key difference is that a single
hierarchical clustering allows construction of models with any number of
states.

CalculateImpliedTimescales.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculates the implied timescales for a python range of MSM lag times.
This allows you to validate whether a given model is Markovian. Notes:

#. You might get a SparseEfficiencyWarning for every lag time. Ignore
   this.

#. Lagtimes are input in units of the time spacing between successive
   trajectory frames. If your trajectories are stored every 10 ns, then
   -l 1,4 estimates implied timescales with lagtimes 10, 20, 30, 40 ns.

PlotImpliedTimescales.py
~~~~~~~~~~~~~~~~~~~~~~~~

A template for generating an implied timescales plot.

BuildMSM.py
~~~~~~~~~~~

Estimate a reversible transition and count matrix using a two step
process:

#. Use Tarjan algorithm to find the maximal strongly-connected (ergodic)
   subgraph

#. Use likelihood maximization to estimate a reversible count matrix
   consistent with your data

This script also outputs the equilibrium populations of the resulting
model, as well as a mapping from the original states to the final
(ergodic) states.

GetRandomConfs.py
~~~~~~~~~~~~~~~~~

Selects random conformations from each state of your MSM. This is very
useful for efficient calculation of observables.

CalculateClusterRadii.py
~~~~~~~~~~~~~~~~~~~~~~~~

Calculates the mean RMSD of all assigned snapshots to their cluster
generator for each cluster. Gives an indication of how structurally
diverse clusters are.

CalculateRMSD.py
~~~~~~~~~~~~~~~~

Calculate the RMSD between a PDB and a trajectory (or set of cluster
centers). Useful for deciding which clusters belong to the folded,
unfolded, or transition state ensembles (or any other grouping!)

CalculateProjectRMSD.py
~~~~~~~~~~~~~~~~~~~~~~~

Calculates the RMSD of all conformations in a project to a given
conformation.

CalculateTPT.py
~~~~~~~~~~~~~~~

Performs Transition Path Theory (TPT) calculations. You will need to
define good starting (reactants/U) and ending (products/F) ensembles for
this script. Writes the forward and backward committors and the net flux
matrix

SavePDBs.py
~~~~~~~~~~~

Allows you to sample random PDBs from particular states and save them to
disk.

PCCA.py
~~~~~~~

Lumps microstates into macrostates using PCCA or PCCA+ . This script
generates a macrostate assignments file from a microstate model.

Notes:

#. We recommend PCCA+ for most applications

#. PCCA+ requires a reversible MSM as input

#. You can discard eigenvectors based on their equilibrium flux
   (fPCCA+).

BACE\_Coarse\_Graining.py
~~~~~~~~~~~~~~~~~~~~~~~~~

An alternative method for lumping microstates into macrostates using a
Bayesian approach (Bayesian agglomerative clustering engine) . This is
an attractive option as it appears to outperform existing spectral
methods. To learn how to use BACE, run the script with the -h option.