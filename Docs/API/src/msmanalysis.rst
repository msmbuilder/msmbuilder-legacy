MSM Analysis: :class:`msmbuilder.msm_analysis`
******

Functions for querying markov state models

Notes
=====
Some functionality was moved to this module from MSMLib in version 2.6

.. currentmodule:: msmbuilder.msm_analysis


Model Queries
=============
.. autosummary::
  :toctree: generated/
  
  sample
  propagate_model
  get_eigenvectors
  get_implied_timescales
  project_observable_onto_transition_matrix
  calc_expectation_timeseries

  
Utils
=====
.. autosummary::
  :toctree: generated/
  
  flatten
  is_transition_matrix
  are_all_dimensions_same
  check_transition
  check_dimensions
  check_for_bad_eigenvalues