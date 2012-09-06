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

