
import commands
import os
import os.path

from fileIO import *

def createJobFile(cmds, fn="pbs_run", name="job", nodes=1, ppn=24, walltime="20:00:00"):
  """Create job file in current directory assuming will run command there.

  ARGUMENTS:
    cmds = commands to run (string)

  OPTIONAL ARGUMENTS:
    fn = job file name (string)
    name = job name (string)
    nodes = number of nodes to run on (int)
    ppn = number of cores to use per node (int)
    walltime = allowed run time for job (string)

  RETURN:
    None
  """

  curDir = os.path.abspath(".")

  content = """#!/bin/bash

#PBS -N %s
#PBS -e %s/run.err
#PBS -o %s/run.out
#PBS -l nodes=%d:ppn=%d
#PBS -l walltime=%s
#PBS -V


PBS_O_WORKDIR='%s'
export PBS_O_WORKDIR
### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------

        echo The master node of this job is `hostname`
        echo The working directory is `echo $PBS_O_WORKDIR`
        echo This job runs on the following nodes:
        echo `cat $PBS_NODEFILE`


### end of information preamble
cd $PBS_O_WORKDIR

# execute commands
%s
""" % (name, curDir, curDir, nodes, ppn, walltime, curDir, cmds)

  writeFile(fn, content)

def submitJob(fn="pbs_run", q=None):
  """Submit job-file to queue q.

  ARGUMENTS:
    None

  OPTIONAL ARGUMENTS:
    fn = name of job file to submit (string)
    q = queue to submit to (string)

  RETURN:
    None
  """

  cmd = "qsub "
  if q != None:
    cmd += "-q %s " % q
  cmd += fn

  out = commands.getoutput(cmd)
  print out

