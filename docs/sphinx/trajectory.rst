Container Classes: :class:`Serializer`, :class:`Conformation`, :class:`Trajectory`
===================================================================================

Trajectory
----------
.. currentmodule:: msmbuilder.Trajectory

.. autoclass:: Trajectory
  :show-inheritance:
  
  Loading from disk
  
  .. autosummary::
    :toctree: generated/
    
    Trajectory.load_trajectory_file
    Trajectory.load_from_pdb
    Trajectory.load_from_pdbList
    Trajectory.load_from_trr
    Trajectory.load_from_dcd
    Trajectory.load_from_xtc
    Trajectory.load_from_hdf
    Trajectory.load_from_lhdf
  
  Reading in chunks
  
  .. autosummary::
    :toctree: generated/
    
    Trajectory.enum_chunks_from_hdf
    Trajectory.enum_chunks_from_lhdf
  
  
  Saving to disk
  
  .. autosummary::
    :toctree: generated/
  
    Trajectory.save
    Trajectory.SaveToHDF
    Trajectory.save_to_lhdf
    Trajectory.save_to_xtc
    Trajectory.save_to_pdb
    Trajectory.save_to_xyz
  
  Read a single frame
  
  .. autosummary::
    :toctree: generated/
  
    Trajectory.read_frame
    Trajectory.read_hdf_frame
    Trajectory.read_lhdf_frame
    Trajectory.read_dcd_frame
    Trajectory.read_xtc_frame

  
  Appending
  
  .. autosummary::
    :toctree: generated/
  
    Trajectory.append_frames_to_file
    Trajectory.AppendPDB
    
  Manipulation
  
  .. autosummary::
    :toctree: generated/
    
    Trajectory.restrict_atom_indices
    Trajectory.subsample
    Trajectory.GetEnumeratedAtomID
    Trajectory.GetEnumeratedResidueID
    Trajectory.GetNumberOfAtoms
    Trajectory.GetNumberOfResidues
    