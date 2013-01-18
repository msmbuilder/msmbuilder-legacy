#!/usr/bin/env python
import sys, os
import numpy as np
import logging
import IPython as ip
from IPython import parallel
from IPython.parallel.error import RemoteError

from msmbuilder import arglib
from msmbuilder import metrics
from msmbuilder import Project

from parallel_assign import remote, local

def setup_logger(console_stream=sys.stdout):
    """
    Setup the logger
    """
    formatter = logging.Formatter('%(name)s: %(asctime)s: %(message)s',
                                  '%I:%M:%S %p')
    console_handler = logging.StreamHandler(console_stream)
    console_handler.setFormatter(formatter)
    logger = logging.getLogger(os.path.split(sys.argv[0])[1])
    logger.root.handlers = [console_handler]
    
    return logger


def main(args, metric, logger):
    
    project = Project.load_from(args.project)
    if not os.path.exists(args.generators):
        raise IOError('Could not open generators')
    generators = os.path.abspath(args.generators)
    output_dir = os.path.abspath(args.output_dir)
    
    # connect to the workers
    try:
        json_file = client_json_file(args.profile, args.cluster_id)
        client = parallel.Client(json_file, timeout=2)
    except parallel.error.TimeoutError as exception:
        msg = '\nparallel.error.TimeoutError: ' + str(exception)
        msg += "\n\nPerhaps you didn't start a controller?\n"
        msg += "(hint, use ipcluster start)"
        print >> sys.stderr, msg
        sys.exit(1)
        
    lview = client.load_balanced_view()
    
    # partition the frames into a bunch of vtrajs
    all_vtrajs = local.partition(project, args.chunk_size)
    
    # initialze the containers to save to disk
    f_assignments, f_distances = local.setup_containers(output_dir,
        project, all_vtrajs)
    
    # get the chunks that have not been computed yet
    valid_indices = np.where(f_assignments.root.completed_vtrajs[:] == False)[0]
    remaining_vtrajs = np.array(all_vtrajs)[valid_indices].tolist()

    logger.info('%d/%d jobs remaining', len(remaining_vtrajs), len(all_vtrajs))
    
    # send the workers the files they need to get started
    # dview.apply_sync(remote.load_gens, generators, project['ConfFilename'],
    #    metric)
    
    # get the workers going
    n_jobs = len(remaining_vtrajs)
    amr = lview.map(remote.assign, remaining_vtrajs,
                    [generators]*n_jobs, [metric]*n_jobs, chunksize=1)
    
    pending = set(amr.msg_ids)
    
    while pending:
        client.wait(pending, 1e-3)
        # finished is the set of msg_ids that are complete
        finished = pending.difference(client.outstanding)
        # update pending to exclude those that just finished
        pending = pending.difference(finished)
        for msg_id in finished:
            # we know these are done, so don't worry about blocking
            async = client.get_result(msg_id)
            
            try:
                assignments, distances, chunk = async.result[0]
            except RemoteError as e:
                print 'Remote Error:'
                e.print_traceback()
                raise
                
            vtraj_id = local.save(f_assignments, f_distances, assignments, distances, chunk)
            
            log_status(logger, len(pending), n_jobs, vtraj_id, async)
                
            
    f_assignments.close()
    f_distances.close()
    
    logger.info('All done, exiting.')

def log_status(logger, n_pending, n_jobs, job_id, async_result):
    """After a job has completed, log the status of the map to the console
    
    Parameters
    ----------
    logger : logging.Logger
        logger to print to
    n_pending : int
        number of jobs still remaining
    n_jobs : int
        total number of jobs in map
    job_id : int
         the id of the job that just completed (between 0 and n_jobs)
    async_esult : IPython.parallel.client.asyncresult.AsyncMapResult
         the container with the job results. includes not only the output,
         but also metadata describing execution time, etc.
    """

    if ip.release.version >= '0.13':
        t_since_submit = async_result.completed - async_result.submitted
        time_remaining = n_pending * (t_since_submit) / (n_jobs - n_pending)
        td  = (async_result.completed - async_result.started)
        #this is equivalent to the td.total_seconds() method, which was
        #introduced in python 2.7
        execution_time = (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / float(10**6)
        eta = (async_result.completed + time_remaining).strftime('%I:%M %p')

    else:
        execution_time, eta = '?', '?'
        
            
    logger.info('engine: %s; chunk %s; %ss; status: %s; %s/%s remaining; eta %s',
                async_result.metadata.engine_id, job_id, execution_time,
                async_result.status, n_pending, n_jobs, eta)


def setup_parser():
    parser = arglib.ArgumentParser("""
Assign data that were not originally used in the clustering (because of
striding) to the microstates. This is applicable to all medoid-based clustering
algorithms, which includes all those implemented by Cluster.py except the
hierarchical methods. (For assigning to a hierarchical clustering, use
AssignHierarchical.py)
    
This code uses IPython.parallel to get parallelism accross many nodes. Consult
the documentation for details on how to run it""", get_metric=True)
    parser.add_argument('project')
    parser.add_argument( dest='generators', help='''Trajectory file containing
        the structures of each of the cluster centers.''')
    parser.add_argument('output_dir')
    parser.add_argument('chunk_size', help='''Number of frames to processes per worker.
        Each chunk requires some communication overhead, so you should use relativly large chunks''',
        default=1000, type=int)
    parser.add_argument('profile', help='IPython.parallel profile to use.', default='default')
    parser.add_argument('cluster_id', help='IPython.parallel cluster_id to use', default='')

    args, metric = parser.parse_args()
    return args, metric


def client_json_file(profile='default', cluster_id=None):
    """
    Get the path to the ipcontroller-client.json file. This really shouldn't be necessary, except that
    IPython doesn't automatically insert the cluster_id in the way that it should. I submitted a pull
    request to fix it, but here is a monkey patch in the mean time
    """
    from IPython.core.profiledir import ProfileDir
    from IPython.utils.path import get_ipython_dir
    
    profile_dir = ProfileDir.find_profile_dir_by_name(get_ipython_dir(), profile)
    if not cluster_id:
        client_json = 'ipcontroller-client.json'
    else:
        client_json = 'ipcontroller-%s-client.json' % cluster_id
    filename = os.path.join(profile_dir.security_dir, client_json)
    if not os.path.exists(filename):
        raise ValueError('controller information not found at: %s' % filename)
    return filename
    
if __name__ == '__main__':
    args, metric = setup_parser()
    logger = setup_logger()
    main(args, metric, logger)
