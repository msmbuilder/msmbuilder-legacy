import os
import glob
import logging
import numpy as np
from msmbuilder import Project, utils, arglib
import mdtraj as md
logger = logging.getLogger('msmbuilder.scripts.RebuildProject')

parser = arglib.ArgumentParser(description="""
Rebuild the project file (ProjectInfo.yaml). This is useful when
trajectory files have been  deleted, or when you have lost your ProjectInfo 
file. \nOutput: ProjectInfo.yaml""")
parser.add_argument('traj_dir', default="./Trajectories/")
parser.add_argument('conf_filename', default="native.pdb")
parser.add_argument('project_filename', default="./ProjectInfo.yaml")
parser.add_argument('iext', default=".h5")


def run(traj_dir, conf_filename, project_filename, iext):
    logger.info("Rebuilding project.")
    file_list = glob.glob(traj_dir + "/trj*%s" % iext)
    num_traj = len(file_list)

    traj_lengths = np.zeros(num_traj, 'int')
    traj_paths = []

    if not os.path.exists(conf_filename):
        raise(IOError("Cannot find conformation file %s" % conf_filename))

    file_list = sorted(file_list, key=utils.keynat)
    for i, filename in enumerate(file_list):
        traj_lengths[i] = len(md.open(filename))
        traj_paths.append(filename)

    records = {
        "conf_filename": conf_filename,
        "traj_lengths": traj_lengths,
        "traj_paths": traj_paths,
        "traj_errors": [None for i in xrange(num_traj)],
        "traj_converted_from": [[] for i in xrange(num_traj)]
    }

    p = Project(records)
    p.save(project_filename)
    logger.info("Wrote %s" % project_filename)

def entry_point():
    args = parser.parse_args()
    run(args.traj_dir, args.conf_filename, args.project_filename, args.iext)

if __name__ == "__main__":
    entry_point()
