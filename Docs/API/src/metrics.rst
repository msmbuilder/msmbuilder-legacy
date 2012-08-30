Distance Metrics: :class:`msmbuilder.metrics`
=============================================

.. currentmodule:: msmbuilder.metrics

Distance metrics in MSMBuilder object which compute the the distance between two frames from an MD simulation. A variety of distance metrics are supported, which all implement the same interface defined in :class:`msmbuilder.metrics.AbstractDistanceMetric`.

The basic architecture of the distance metrics are that all of their configurables are set in an __init__() method which is specific to each metric. A method called :class:`prepare_trajectory(self, trajectory)` takes as input a :class:`msmbuilder.Trajectory.Trajectory` object and returns a python container object (at this point either a custom container supporting slice operations for RMSD or a numpy array for every other metric). This method does any required preprocessing such as extracting the dihedral angles for the :class:`msmbuilder.metrics.Dihedral` metric. Then, distances can be computed using the metric object and the prepared trajectory by invoking a variety of methods on the metric object with the prepared trajectories, e.g. :class:`one_to_all(self, ptraj1, ptraj2, index1)` to compute the distance from one frame to every frame in a trajectory, :class:`one_to_many(self, ptraj1, ptraj2, index1, indices2)` to compute the distance from one frame to a collection of frames in a trajectory, or other similar metrics.

Currently, the following distance metrics are implemented:

:class:`msmbuilder.metrics.RMSD` Measures the cartesians room mean square deviation between corresponding atoms in two frames after rotating and translating the two frames to bring them into maximum coincidence.

:class:`msmbuilder.metrics.Dihedral` Measures the difference in the dihedral angles between two frames. Because of the periodic symmetry of dihedral angles, this is implemented by comparing difference in the sine and cosine of each the dihedral angles, which is equivalent to the difference of the angles in the complex plane.

:class:`msmbuilder.metrics.ContinuousContact` Each frame is represented by the pairwise distance between residues, and then frames are compared based on the difference between these sets of distances. The set of distances that are monitored is configurable. The sense in which the distance between residues is computed (C-alpha, closest heavy atom, etc) is configurable.

:class:`msmbuilder.metrics.BooleanContact` Each frame is represented a set of booleans representing whether each of a set of residue-residue contacts is present or absent, and frames are compared based on the different between these sets of booleans.

:class:`msmbuilder.metrics.AtomPairs` While ContinuousCotact monitors the distance between `residues`, AtomPairs monitors the distance between specific atoms. Each frame is represented by the pairwise distance between a set of atoms, and the distance between frames is computed based on the the distance between these vectors of pairwise distances.

:class:`msmbuilder.metrics.Hybrid` This class can be used to compose metrics additively. For example, you can have a metric that is 0.5 * rmsd + 0.5 * dihedral. Note that the different metrics are probably in different units, so you'll have to be careful what weights you want to give them when you add them.

:class:`msmbuilder.metrics.HybridPNorm` This metric is used to compose other metrics by adding them in quadrature (p=2) or some other power mean. It is more general than :class:`msmbuilder.metrics.Hybrid`, which is a special case for p=1.

:class:`msmbuilder.metrics.Rg` This just compares frames based on the difference in their Rg (one number).

Performance Notes
-----------------

Most of the leg work for these distance metrics is done in C with shared memory parallelism (OpenMP) to take advantage of multicore architecture. The RMSD code has been highly optimized using SSE intrinsics, attention to cache performance, a faster matrix multiply for the Nx3 case than BLAS, etc. The other codes have been less highly optimized, but are algorithmically simpler than RMSD.

If you'd like to reduce the number of cores used by the distance metrics, you can set the OMP_NUM_THREADS environment variable. (e.g. `export OMP_NUM_THREADS=1` in bash to use only 1 core/thread).

Kinetic Distance Metrics
------------------------

New distance metrics for kinetic clustering are currently an active area of research in the Pande Lab.

Concrete Metrics
---------------------

RMSD
~~~~

.. autoclass:: RMSD
  :show-inheritance:
  
  .. autosummary::
    :toctree: generated/
  
    RMSD.__init__
    RMSD.prepare_trajectory
    RMSD.one_to_many
    RMSD.one_to_all
    RMSD.all_pairwise


Vectorized Metrics
~~~~~~~~~~~~~~~~~~

.. autoclass:: Dihedral
  :show-inheritance:

  .. autosummary::
    :toctree: generated/ 

    Dihedral.__init__
    Dihedral.prepare_trajectory
    Dihedral.one_to_all
    Dihedral.one_to_many
    Dihedral.many_to_many
    Dihedral.all_pairwise
    Dihedral.all_to_all

.. autoclass:: ContinuousContact
  :show-inheritance:

  .. autosummary::
    :toctree: generated/ 

    ContinuousContact.__init__
    ContinuousContact.prepare_trajectory
    ContinuousContact.one_to_all
    ContinuousContact.one_to_many
    ContinuousContact.many_to_many
    ContinuousContact.all_pairwise
    ContinuousContact.all_to_all

.. autoclass:: msmbuilder.metrics.BooleanContact
  :show-inheritance:

  .. autosummary::
    :toctree: generated/ 

    BooleanContact.__init__
    BooleanContact.prepare_trajectory
    BooleanContact.one_to_all
    BooleanContact.one_to_many
    BooleanContact.many_to_many
    BooleanContact.all_pairwise
    BooleanContact.all_to_all
    
    
.. autoclass:: msmbuilder.metrics.AtomPairs
  :show-inheritance:
  
  .. autosummary::
    :toctree: generated/ 

    AtomPairs.__init__
    AtomPairs.prepare_trajectory
    AtomPairs.one_to_all
    AtomPairs.one_to_many
    AtomPairs.many_to_many
    AtomPairs.all_pairwise
    AtomPairs.all_to_all

.. autoclass:: msmbuilder.metrics.Rg
  :show-inheritance:

  .. autosummary::
    :toctree: generated/ 

    Rg.__init__
    Rg.prepare_trajectory
    Rg.one_to_all
    Rg.one_to_many
    Rg.many_to_many
    Rg.all_pairwise
    Rg.all_to_all
  

Combination Metrics
~~~~~~~~~~~~~~~~~~~

.. autoclass:: msmbuilder.metrics.Hybrid

.. autoclass:: msmbuilder.metrics.HybridPNorm

Abstract Classes
----------------

.. autoclass:: AbstractDistanceMetric

  .. autosummary::
    :toctree: generated/

    AbstractDistanceMetric.prepare_trajectory
    AbstractDistanceMetric.all_pairwise
    AbstractDistanceMetric.one_to_all
    AbstractDistanceMetric.one_to_many

.. autoclass:: Vectorized

  .. autosummary::
    :toctree: generated/

    Vectorized.__init__
    Vectorized.prepare_trajectory
    Vectorized.all_pairwise
    Vectorized.all_to_all
    Vectorized.many_to_many
    Vectorized.one_to_all
    Vectorized.one_to_many


Utility Methods
---------------

.. autosummary::
  :toctree: generated/

  fast_cdist
  fast_pdist
