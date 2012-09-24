.. currentmodule:: msmbuilder.cfep

Cut Based Free Energy Profile :mod:`msmbuilder.cfep`
====================================================

Code for computing cut-based free energy profiles, and optimal
reaction coordinates within that framework. This code was 
contributed generously by Sergei Krivov, with optimizations
and modifications by TJ Lane.

This code employs `scipy.weave`, so g++ and OpenMP are
required. Most machines should have this functionality by default.

Optimization
------------

.. autoclass:: CutCoordinate

  .. autosummary::
    :toctree: generated/
    
    CutCoordinate.evaluate_partition_functions
    CutCoordinate.plot
    CutCoordinate.reaction_mfpt
    CutCoordinate.rescale_to_natural_coordinate
    CutCoordinate.set_coordinate_as_committors
    CutCoordinate.set_coordinate_as_eigvector2
    CutCoordinate.set_coordinate_values
    
.. autoclass:: VariableCoordinate

  .. autosummary::
    :toctree: generated/
    
    VariableCoordinate.optimize

Example Coordinate
--------------------
.. autosummary:: 
   :toctree: generated/

   contact_reaction_coordinate
