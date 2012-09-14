.. module:: msmbuilder.MSMLib

MSM Construction: :mod:`msmbuilder.MSMLib`
============================================

Functions for building MSMs

Notes
-----
* Assignments typically refer to a numpy array of integers such that Assignments[i,j] gives the state of trajectory i, frame j.
* Transition and Count matrices are typically stored in scipy.sparse.csr_matrix format.
* Some functionality from this module was moved into msmanalysis in version2.6
  
Mapping
-------

.. autosummary::
  :toctree: generated/
  
  apply_mapping_to_assignments
  apply_mapping_to_vector
  renumber_states
  invert_assignments
  
  
Trimming
--------

.. autosummary::
  :toctree: generated/

  tarjan
  ergodic_trim

Model Building
--------------

.. autosummary::
  :toctree: generated/

  build_msm
  estimate_rate_matrix
  mle_reversible_count_matrix
  estimate_transition_matrix
  get_count_matrix_from_assignments
  get_counts_from_traj
  log_likelihood
