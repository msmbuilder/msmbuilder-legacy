.. module:: msmbuilder.clustering

Clustering: :class:`msmbuilder.clustering`
==========================================


MSMBuilder uses clustering on MD trajectories to discretize phase space. A number of clustering algorithms are provided, and each can be used with a variety of metrics (link to metrics page?) to produce a large set of possible discretizations.

Currently, the following clustering algorithms are available

:class:`KCenters`, :class:`HybridKMedoids`, :class:`Clarans`, :class:`Hierarchical`

Abstract Classes
----------------

.. autoclass:: BaseFlatClusterer

  .. autosummary::
    :toctree: generated/
    
    BaseFlatClusterer.get_distances
    BaseFlatClusterer.get_assignments
    BaseFlatClusterer.get_generators_as_traj

Flat Clustering Classes
-----------------------

.. autoclass:: KCenters
  :show-inheritance:
  
  .. autosummary::
    :toctree: generated/
    
    KCenters.__init__
    KCenters.get_distances
    KCenters.get_assignments
    KCenters.get_generators_as_traj
    
.. autoclass:: HybridKMedoids
  :show-inheritance:

  .. autosummary::
    :toctree: generated/

    HybridKMedoids.__init__
    HybridKMedoids.get_distances
    HybridKMedoids.get_assignments
    HybridKMedoids.get_generators_as_traj

.. autoclass:: Clarans
  :show-inheritance:

  .. autosummary::
    :toctree: generated/

    Clarans.__init__
    Clarans.get_distances
    Clarans.get_assignments
    Clarans.get_generators_as_traj

.. autoclass:: SubsampledClarans
  :show-inheritance:

  .. autosummary::
    :toctree: generated/

    SubsampledClarans.__init__
    SubsampledClarans.get_distances
    SubsampledClarans.get_assignments
    SubsampledClarans.get_generators_as_traj
    
    
Hierarchical Clustering
-----------------------

.. autoclass:: Hierarchical
  
  .. autosummary::
    :toctree: generated/

    Hierarchical.get_assignments 
    Hierarchical.load_from_disk
    Hierarchical.save_to_disk

Clustering Functions
--------------------

.. autosummary::
  :toctree: generated/
  
  _kcenters
  _hybrid_kmedoids
  _clarans

Utility Functions
-----------------

.. autosummary::
  :toctree: generated/

  _assign
  concatenate_trajectories
  unconcatenate_trajectory
  split
  stochastic_subsample
  deterministic_subsample
  empty_trajectory_like
  p_norm


