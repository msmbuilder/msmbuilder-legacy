import numpy as np
from msmbuilder import Trajectory, Project, utils
import glob

traj_dir = "./Trajectories/"
conf_filename = "native.pdb"
project_filename = './ProjectInfo.yaml'

file_list = glob.glob(traj_dir + "/trj*.lh5")
num_traj = len(file_list)

traj_lengths = np.zeros(num_traj,'int')
traj_paths = []
traj_converted_from = []

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
