FahProject: :class:`msmbuilder.FahProject`
==========================================

Interfacing with Folding@Home projects
--------------------------------------

.. currentmodule:: msmbuilder.FahProject

.. autoclass:: FahProject
  
  .. autosummary::
    :toctree: generated/
    
    FahProject.__init__
    FahProject.restart_server
    FahProject.send_error_email
    FahProject.save_memory_state
    FahProject.load_memory_state
    
  injection methods: Although these methods are documented as being in a separate class :class:`_inject`, they are actually are available inside FahProject under :class:`FahProject.inject` ::
  >> fp = FahProject(mypdb, myprojectnumber)
  >> fp.inject.set_project_basepath(...)

  .. autosummary::
    :toctree: generated/
    
    _inject.set_project_basepath
    _inject.new_run
    _inject.stop_clone
    
  retreival methods: Although these methods are documented as being in a separate class :class:`_inject`, they are actually are available inside FahProject under :class:`FahProject.retrieve` ::
  >> fp = FahProject(mypdb, myprojectnumber)
  >> fp.retrieve.write_all_trajectories(...)
    
  .. autosummary::
    :toctree: generated/

    _retrieve.write_all_trajectories
    _retrieve.update_trajectories
    _retrieve.write_trajectory
                    