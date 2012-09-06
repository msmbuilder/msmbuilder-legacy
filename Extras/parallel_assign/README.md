 assignment for MSMBuilder using IPython.parallel 
=========================================================

Overview
--------
This package provides parallel assignment for MSMBuilder.

Because it uses `IPython.parallel` as opposed to my homebuilt mpi4py
code, this code is a lot more flexible. It also should be a lot more stable/bug
free. It checkpoints much faster by only writing the necessary incremental
updates to the files on disk as opposed to completely rewriting them each time.

Simple PBS Script
-----------------

Here's an example of how to use this code in a PBS job

    #PBS -N assign
	#PBS -l walltime=1:00:00
	#PBS -l nodes=2:ppn=24
	#PBS -q default
	#PBS -o /dev/null
	#PBS -e /dev/null 

    PROJECT=$HOME/WW/ProjectInfo.h5
	GENS=$HOME/WW/rmsd/hybrid5k/Gens.lh5
	OUT_DIR=$HOME/test
	CHUNK_SIZE=1000
	METRIC="rmsd -a $HOME/WW/AtomIndices.dat"
	LOG_FILE=$OUT_DIR/assign.log

    #make sure the outdir exists
	mkdir -p $OUT_DIR

    #set 11 threads so that one is available for AssignIPP and ipcontroller
	export OMP_NUM_THREADS=11

    if [ -n "$PBS_ENVIRONMENT" ]; then
        # execute inside PBS environment
		cd $PBS_O_WORKDIR

        ipcontroller --ip='*' --cluster-id $PBS_JOBID &> $LOG_FILE.ipcontroller &
		sleep 2

        mpirun --npernode 1 -np $PBS_NUM_NODES --machinefile $PBS_NODEFILE ipengine --cluster-id $PBS_JOBID &> $LOG_FILE.mpirun &
		sleep 5 # leave enough time for the engines to connect to the controller

        AssignIPP.py -p $PROJECT -g $GENS -o $OUT_DIR -c $CHUNK_SIZE -C $PBS_JOBID $METRIC &> $LOG_FILE
	else

        # we're not executing in a PBS environment, but we want
		# to test this script anyways
		CLUSTER_ID=$0

        ipcontroller --ip='*' --cluster-id $CLUSTER_ID & # &> $LOG_FILE.ipcontroller &
		sleep 5
		ipengine --cluster-id $CLUSTER_ID & # &> $LOG_FILE.mpirun &
		sleep 5
		AssignIPP.py -p $PROJECT -g $GENS -o $OUT_DIR -c $CHUNK_SIZE -C $CLUSTER_ID $METRIC # &> $LOG_FILE

        kill %1 %2 %3
	fi

    echo "finished"


Workflow
--------
The flexibility of `IPython.parallel` comes at slight cost in complexity.
`IPython.parallel` is basically a master-worker framework. The first step
is thus to start workers.

To start 1 worker on your local node, you can use the command

    ipcluster start --n=1

The `ipcluster` command does two things. First, it starts something called a
controller, and next it starts a bunch of engines (in this case only one since
we gave it n=1). The engines' job is to actually do our work. The controller is
not as interested. Its job is to coordinate with the engines and handle things
like task scheduling.

Once the controller/engines are up and running, we can run `AssignIPP.py`.
One of the first things the script will try to do is connect to the controller.
Then it will prepare the jobs, submit then, and save the results as they return.

The Script
----------

`AssignIPP.py` options, that you can see with `AssignIPP.py -h`. A new option
which deserves some explanation is the chunk_size option.

Previous assignment codes that were written processed each trajectory individually.
This is not necessarily the best choice. If you have a lot of short trajectories,
processing each trajectory individually leads to a lot of overhead that you
don't need. On the other hand, if you only have a single long trajectory,
you have no way to do the computation in parallel over multiple nodes unless you
split it up.

`AssignIPP.py` partitions the frames in your trajectory into a bunch of chunks.
Each chunk may incorporate frames from a single trajectory or multiple trajectories.
These chunks are then the unit of the parallelism between nodes. When the assignment
is completed for each chunk, those results are immediately written to disk.

The size of the chunks is set from the command line with chunk_size. Using large
chunks will reduce the communication overhead, but if you have fewer chunks
than you do engines, not all the engines will actually be able to do anything.
Using small chunks will also lead to more frequent checkpointing.


If you start up `AssignIPP.py` pointing to an output directory that already
contains results (i.e. an Assignments.h5 and Assignments.h5.distances), it will
pick up where it left off. The only caveat is that you need to supply an identical
chunk_size as you did previously.

PBS Workers
-----------

Simply starting a single engine on your local node is pretty boring. Instead, you
probably want your engine to be on other nodes.

On Stanford's certainty cluster, I can do the following from two DIFFERENT nodes
    
    rmcgibbo@certainty-a:
    $ ipcontroller --ip '*'
    
    rmcgibbo@certainty-b:
    $ ipengine
    
On certainty-a, I see

    2012-08-17 00:14:07.051 [IPControllerApp] registration::finished registering engine 0:'661ecce8-8d2e-41d2-97f4-6715fe4a8692'
    2012-08-17 00:14:07.052 [IPControllerApp] engine::Engine Connected: 0
    
which indicates that the engine on one node has successfully connected to
the controller on a different node.

The flag `--ip='*` indicates that we're allowing engines to connect from any ip
address, which is necessary.

The `ipcluster` command, which calls both `ipcontroller` and `ipengine`, has a
very convenient feature -- profiles, which helps to automate all of this setup.
Robert put a few profiles online at https://github.com/rmcgibbo/ipython_parallel_profiles

If you download and install that package, you can run

    ipcluster start --profile=pbs1hr --n 10
    
which will start a controller on the local machine and then submit a PBS job to the
default queue asking for ten nodes for an hour each. Each of those jobs will execute
the `ipengine` command to connect to the controller, and then you're in business.

Then you can run `AssignIPP.py` on your local node, and it will connect to the
processes in the PBS queue and they will do the work.

SSH Workers
-----------

We're still exploring this, but I've got the following to work

    rmcgibbo@vspm42-ubuntu ~
    $ ipcontroller --ip='*'

    rmcgibbo@vsp-compute-01 ~
    $ scp vspm42-ubuntu:/home/rmcgibbo/.config/ipython/profile_default/security/ipcontroller-engine.json .
    rmcgibbo@vsp-compute-01 ~
    $ ipengine --file=ipcontroller-engine.json
    
Resources
---------

http://ipython.org/ipython-doc/dev/parallel/parallel_process.html



