.. module:: msmbuilder.tpt

Transition Path Theory : :mod:`msmbuilder.tpt`
==============================================

Functions for performing Transition Path Theory calculations.

Path Finding
-------------
.. autosummary::
  :toctree: generated/
  
  find_top_paths
  Dijkstra
  find_path_bottleneck
  calculate_fluxes
  calculate_net_fluxes
  
  
MFPT & Committor Finding Functions
----------------------------------
.. autosummary::
  :toctree: generated/

  calculate_ensemble_mfpt
  calculate_avg_TP_time
  calculate_mfpt
  calculate_all_to_all_mfpt
  calculate_committors
  
Hub Scores, Conditional Committors and Related Quantities
---------------------------------------------------------
.. autosummary::
  :toctree: generated/

  calculate_fraction_visits
  calculate_hub_score
  calculate_all_hub_scores