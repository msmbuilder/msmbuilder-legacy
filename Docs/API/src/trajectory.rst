Container Classes: :class:`Serializer`, :class:`Conformation`, :class:`Trajectory`
===================================================================================

Serializer
----------
.. currentmodule:: msmbuilder.Serializer

.. class:: Serializer (DictLikeObject={})
  
  A generic class for dumping dictionaries of data onto disk using the pytables HDF5 library. 
  
  .. autosummary::
      :toctree: generated/

      Serializer.__init__
      Serializer.CheckIfFileExists
      Serializer.LoadCSRMatrix
      Serializer.LoadFromHDF
      Serializer.SaveToHDF
      Serializer.SaveEntryAsEArray
      Serializer.SaveEntryAsCArray
      Serializer.SaveCSRMatrix
      Serializer.LoadCSRMatrix
      Serializer.SaveData
      Serializer.LoadData

Trajectory
----------
.. currentmodule:: msmbuilder.Trajectory

.. class:: Trajectory (S)

  Represent a sequence of conformations
  
  Loading from disk
  
  .. autosummary::
    :toctree: generated/
    
    Trajectory.LoadTrajectoryFile
    Trajectory.LoadFromPDB
    Trajectory.LoadFromPDBList
    Trajectory.LoadFromDCD    
    Trajectory.LoadFromTRR
    Trajectory.LoadFromHDF
    Trajectory.LoadFromXTC
    Trajectory.LoadFromLHDF
  
  Saving to disk
  
  .. autosummary::
    :toctree: generated/
  
    Trajectory.Save
    Trajectory.SaveToHDF
    Trajectory.SaveToLHDF
    Trajectory.SaveToXTC
    Trajectory.SaveToPDB
    Trajectory.SaveToXYZ
  
  Read a single frame
  
  .. autosummary::
    :toctree: generated/
  
    Trajectory.ReadDCDFrame
    Trajectory.ReadLHDF5Frame
    Trajectory.ReadFrame
    Trajectory.ReadXTCFrame
    Trajectory.ReadHDF5Frame
  
  Appending
  
  .. autosummary::
    :toctree: generated/
  
    Trajectory.AppendFramesToFile
    Trajectory.AppendPDB
    
  Manipulation
  
  .. autosummary::
    :toctree: generated/
    
    Trajectory.GetEnumeratedAtomID
    Trajectory.GetEnumeratedResidueID
    Trajectory.GetNumberOfAtoms
    Trajectory.GetNumberOfResidues
    