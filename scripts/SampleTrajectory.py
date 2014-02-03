import numpy as np
from mdtraj import io
from msmbuilder import arglib
from msmbuilder import msm_analysis
from msmbuilder import Project
import scipy.io
import logging
logger = logging.getLogger('msmbuilder.scripts.SampleTrajectory')
DEBUG = True

parser = arglib.ArgumentParser(description="""
Create an MSM movie by sampling a sequence of states and sampling a 
random conformation from each state in the sequence.  
""")
parser.add_argument('project')
parser.add_argument('assignments', default='Data/Assignments.Fixed.h5')
parser.add_argument('tprob', default='Data/tProb.mtx')
parser.add_argument('num_steps')
parser.add_argument('starting_state', type=int,
                    help='''Which state to start trajectory from.''')
parser.add_argument('output', default='sample_traj.pdb',
                    help="""The filename of your output trajectory.  The filetype suffix will be used to select the output file format.""")


def main():
    """Parse command line inputs, load up files, and build a movie."""
    args = parser.parse_args()
    try:
        assignments = io.loadh(args.assignments, 'arr_0')
    except KeyError:
        assignments = io.loadh(args.assignments, 'Data')

    num_steps = int(args.num_steps)
    starting_state = int(args.starting_state)

    project = Project.load_from(args.project)
    T = scipy.io.mmread(args.tprob).tocsr()

    state_traj = msm_analysis.sample(T, starting_state, num_steps)
    sampled_traj = project.get_random_confs_from_states(
        assignments, state_traj, 1)
    traj = sampled_traj[0]
    traj["XYZList"] = np.array([t["XYZList"][0] for t in sampled_traj])
    traj.save(args.output)

if __name__ == '__main__':
    main()
