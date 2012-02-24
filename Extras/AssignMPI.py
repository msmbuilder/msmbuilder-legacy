#!/usr/bin/python
# This file is part of MSMBuilder.
#
# Copyright 2011 Stanford University
#
# MSMBuilder is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#

""" 
Robert McGibbon
Stanford University
Pande Lab
July 2011

Assign data to generators that was not originally
included in clustering, in parallel via mpi and dtm

Invoke the program with mpirun -n <NUM PROCS> python AssignMPI.py [options]

Dependencies:
deap
mpi4py
"""

import os
import sys
import numpy as np

try:
    from deap import dtm
except:
    print "You must install the packages deap and mpi4py to run this script"
    raise

from msmbuilder import Project, Serializer, Trajectory
import ArgLib

def run(project_file, generators_file, atom_indices_file,
        assignments_file, rmsd_file, which_traj_file, num_chunks):
    """
    Run assignments to generators in paralell

    project_file: path to ProjectInfo file
    generators_file: path to generators file
    atom_indices_file: path to atom indices file

    assigments_file: path to place assignments OUTPUT
    rmsd_file: path to place assignments' rmsd OUTPUT
    which_traj_file: path to place assignments' which_traj OUTPUT
    """
    
    # load into memory
    project = Project.Project.LoadFromHDF(project_file)
    generators = Trajectory.Trajectory.LoadTrajectoryFile(generators_file,
                                                          Conf = project.Conf)
    atom_indices = np.loadtxt(atom_indices_file, int)

    if num_chunks > project['NumTrajs']:
        print >> sys.stderr, "You requested too many chunks. There must not"
        print >> sys.stderr, "be more chunks than there are trajectories."
        print >> sys.stderr, "Exiting"
        sys.exit(1)
    if num_chunks <= 0:
        print >> sys.stderr, "You request too few chunks. There must not "
        print >> sys.stderr, "be zero or fewer chunks."
        print >> sys.stderr, "Exiting"
        sys.exit(1)
        

    # generate the list of jobs
    trajectory_indicies = np.arange(project['NumTrajs'])

    # DEBUGGING:
    # Reduce trajectory_indicies so the job is smaller
    # trajectory_indicies = np.arange(min(3, project['NumTrajs']))

    jobs = []
    for i in xrange(num_chunks):
        which_trajectories = trajectory_indicies[i::num_chunks]
        jobs.append({'which_trajectories' : which_trajectories,
                     'generators': generators,
                     'atom_indices': atom_indices,
                     'project': project})

    # map them out
    print 'Mapping out assignment jobs'
    job_results = dtm.map(run_assign_job, jobs)
    print 'Results recieved. Reducing them down'

    reduce_results(job_results, assignments_file, rmsd_file, which_traj_file)
    return

    
def reduce_results(job_results, assignments_file, rmsd_file, which_traj_file):
    """Take the array of results from each independent jobs
    from the map step and reduce them
    into a single set of results. Then print those results to file"""
    
    # reduce the results
    assignments_list = []
    rmsd_list = []
    which_trajectories_list = []
    for job_result in job_results:
        assignments_list.append(job_result['assigments'])
        rmsd_list.append(job_result['rmsd'])
        which_trajectories_list.append(job_result['which_trajectories'])

    all_assignments, all_rmsd, all_trajectories = Project.MergeMultipleAssignments(assignments_list,
                                                                                   rmsd_list,
                                                                                   which_trajectories_list)

    # output results
    Serializer.SaveData(assignments_file, all_assignments)
    Serializer.SaveData(rmsd_file, all_rmsd)
    Serializer.SaveData(which_traj_file, all_trajectories)

    print "Saved Assignments! Exiting"
    return
    
        
def run_assign_job(job):
    """
    This function is called by map() and spits off a Assign job for
    a set of trajectories

    job needs to be a dict with keys 'which_trajectories', 'generators',
    'project' and 'atom_indices'
    """
    
    try:
        which_trajectories = job['which_trajectories']
        generators = job['generators']
        atom_indices = job['atom_indices']
        project = job['project']
    except KeyError:
        print >> sys.stderr, "KeyError: You're calling assign_job incorrectly."
        print >> sys.stderr, "Exiting"
        sys.exit(1)
    
    assignments, rmsd, which_trajectories = project.AssignProject(generators,
                                                                  AtomIndices=atom_indices,
                                                                  WhichTrajs=which_trajectories)
    return {'assigments': assignments,
            'rmsd': rmsd,
            'which_trajectories': which_trajectories}


def parse():
    print """\nAssigns data to generators that was not originally used in the
clustering. Does this in parallel across many nodes using MPI. Run this script
on a cluster with the following command:

mpirun -n <NUM PROCS> python AssignMPI.py -P <NUM PROCS> [options]

Output:
-- Assignments.h5: a matrix of assignments where each row is a vector
   corresponding to a data trajectory. The values of this vector are the cluster
   assignments.
-- Assignments.h5.RMSD: Gives the RMSD from the assigned frame to its Generator.
-- Assignments.h5.WhichTrajs: Indicates which trajectories were assigned to which proc.
\n"""

    arglist=["projectfn", "generators", "atomindices", "outdir", "procs"]
    options=ArgLib.parse(arglist)
    print sys.argv

    # generate output file paths
    assignments_file = os.path.join(options.outdir, 'Assignments.h5')
    rmsd_file        = os.path.join(options.outdir, 'Assignments.h5.RMSD')
    which_traj_file  = os.path.join(options.outdir, 'Assignments.h5.WhichTrajs')

    # check output file paths
    if os.path.exists(assignments_file):
        print >> sys.stderr, "%s already present in output directory" % assignments_file
        print >> sys.stderr, "Exiting"
        sys.exit(1)
    if os.path.exists(rmsd_file):
        print >> sys.stderr, "%s already present in output directory" % rmsd_file
        print >> sys.stderr, "Exiting"
        sys.exit(1)
    if os.path.exists(which_traj_file):
        print >> sys.stderr, "%s already present in output directory" % which_traj_file
        print >> sys.stderr, "Exiting"
        sys.exit(1)

    return (options, assignments_file, rmsd_file, which_traj_file)


def print_help():
    """ If command line help is specified, print and exit """
    (options, assignments_file, rmsd_file, which_traj_file) = parse()
    sys.exit(1)


def main():
    """ This function is called if we actually decide to execute """
    (options, assignments_file, rmsd_file, which_traj_file) = parse()

    run(project_file      = options.projectfn,
        generators_file   = options.generators,
        atom_indices_file = options.atomindices,
        assignments_file  = assignments_file,
        rmsd_file         = rmsd_file,
        which_traj_file   = which_traj_file,
        num_chunks        = options.procs)
    

if __name__ == "__main__":
    # We don't want to boot the MPI interface if help is requested
    if sys.argv[1] == '-h': print_help()
    elif sys.argv[1] == '--help': print_help()
    else: dtm.start(main)

    
    


