.. currentmodule:: msmbuilder.lumping

Lumping: :class:`msmbuilder.lumping`
============================================

Lumping, or coarse-graining, refers to the processing of creating a few-state model from a many-state model (lumping states together) in such a way that the lumped model can capture the long timescale dynamics (i.e. joining together states which interconvert rapidly)

Core Lumping Routines
---------------------
.. autosummary::
  :toctree: generated/
  
  PCCA
  pcca_plus
  
Private PCCA+ Implementation Code
---------------------------------
.. autosummary::
  :toctree: generated/
  
  has_constraint_violation
  fill_A
  index_search
  get_maps
  objective
  to_square
  to_flat

Helper Functions
----------------
.. autosummary::
  :toctree: generated/
  
  NormalizeLeftEigenvectors
  trim_eigenvectors_by_flux
  