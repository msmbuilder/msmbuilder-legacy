#!/usr/bin/env python
import numpy as np
import glob
import logging
from msmbuilder import Trajectory, Project, utils, arglib

logger = logging.getLogger('msmbuilder.scripts.RebuildProject')


def run(traj_dir, conf_filename, project_filename):
    logger.info("Rebuilding project.")
    file_list = glob.glob(traj_dir + "/trj*.lh5")
    num_traj = len(file_list)
    
    traj_lengths = np.zeros(num_traj,'int')
    traj_paths = []
    
    file_list = sorted(file_list, key=utils.keynat)
    for i,filename in enumerate(file_list):
        traj_lengths[i] = Trajectory.load_trajectory_file(filename,JustInspect=True)[0]
        traj_paths.append(filename)    
    
    records = {
    "conf_filename":conf_filename,
    "traj_lengths":traj_lengths,
    "traj_paths":traj_paths,
    "traj_errors": [None for i in xrange(num_traj)],
    "traj_converted_from":[[] for i in xrange(num_traj)]           
    }
    
    p = Project(records)
    p.save(project_filename)
    logger.info("Wrote %s" % project_filename)
    
if __name__ == "__main__":
    parser = arglib.ArgumentParser(description=
                                   """Rebuild the project file (ProjectInfo.yaml).
                                   This is useful when trajectory files have been 
                                   deleted, or when you have lost your ProjectInfo 
                                   file.
                                   \nOutput: ProjectInfo.yaml
                                   """)
    parser.add_argument('traj_dir',default="./Trajectories/")
    parser.add_argument('conf_filename',default="native.pdb")
    parser.add_argument('project_filename',default="./ProjectInfo.yaml")
    args = parser.parse_args()
    
    run(args.traj_dir, args.conf_filename, args.project_filename)
