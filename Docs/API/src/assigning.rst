.. currentmodule:: msmbuilder.assigning
assigning: :mod:`msmbuilder.assigning`
======================================

This module contains codes for assigning all of the frames in a project
to a set of already-identified cluster centers. This is used when, due to
memory or CPU constraints, the initial clustering step is done with
only a subset of the total data.


Functions
---------

.. autosummary::
  :toctree: generated/
  
  assign_in_memory
  assign_with_checkpoint
  streaming_assign_with_checkpoint