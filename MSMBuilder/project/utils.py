from __future__ import print_function, absolute_import, division
import os
import logging
from msmbuilder.utils import keynat
import tables
import mdtraj as md

from .project import Project

logger = logging.getLogger(__name__)


def get_project_object(traj_directory, conf_filename, out_filename=None):
    """
    This function constructs a msmbuilder.Project object 
    given a directory of trajectories saved as .lh5's. 

    Note that this is only really necessary when a script
    like ConvertDataToLHDF.py converts the data but fails
    to write out the ProjectInfo.yaml file.

    This function can also be used to combine two projects
    by copying and renaming the trajectories in a new 
    folder. Though, it's probably more efficient to just
    do some bash stuff to cat the ProjectInfo.yaml's 
    together and rename the trajectories.

    Inputs:
    -------
    1) traj_directory : directory to find the trajectories
    2) conf_filename : file to find the conformation
    3) out_filename [ None ] : if None, then this function 
        does not save the project file, but if given, the
        function will save the project file and also
        return the object

    Outputs:
    -------
    project : msmbuilder.Project object corresponding to 
        your project.
    """
    # relative to the traj_directory
    traj_paths = sorted(os.listdir(traj_directory), key=keynat)
    # relative to current directory
    traj_paths = [os.path.join(traj_directory, filename) for filename in traj_paths]

    traj_lengths = []

    for traj_filename in traj_paths:  # Get the length of each trajectory
        logger.info(traj_filename)

        if traj_filename.split('.')[-1] in ['hdf', 'h5', 'lh5']:
            with tables.openFile(traj_filename) as f:
                traj_lengths.append(f.root.coordinates.shape[0])

        else:
            traj_lengths.append(md.load(traj_filename).n_frames)

    project = Project({'conf_filename': conf_filename,
                       'traj_lengths': traj_lengths,
                       'traj_paths': traj_paths,
                       'traj_errors': [None] * len(traj_paths),
                       'traj_converted_from': [[None]] * len(traj_paths)})

    if out_filename is None:
        return project
    else:
        project.save(out_filename)
        logger.info('Saved project file to %s', out_filename)
        return project
