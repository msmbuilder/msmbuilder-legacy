"""
Functions that execute remotely on the workers

Note that there are thre globals on each worker, pgens, conf and metric. Also,
due to the way that IPython.parallel works, we do imports inside the functions


"""
PREPARED, PGENS, CONF, METRIC = False, None, None, None

def load_gens(gens_fn, conf_fn, metric):
    """Setup a worker by adding pgens to its global namespace
    
    This is necessary because pgens are not necessarily picklable, so we can't
    just prepare them on the master and then push them to the remote workers --
    instead we want to actually load the pgens from disk and prepare them on
    the remote node
    """
    from msmbuilder import Trajectory
    
    global PGENS, CONF, METRIC, PREPARED
    
    METRIC = metric
    CONF = Trajectory.LoadTrajectoryFile(conf_fn)
    gens = Trajectory.LoadTrajectoryFile(gens_fn)
    PGENS = metric.prepare_trajectory(gens)
    PREPARED = True
    

def assign(vtraj, gens_fn, metric):
    """
    Assign a VTraj to the generators
    
    This executes on the remote workers. It uses two global variables which
    are worker-local
    
    Parameters
    ----------
    vtraj : VTraj
        A list of tuples like (traj_index, slice(start, end))
    
    
    Globals
    -------
    conf : msmbuilder.Trajectory
    metric : msmbuilder.metrics.AbstractDistanceMetric
    
    """
    import numpy as np
    global CONF
    
    if not PREPARED:
        load_gens(gens_fn, vtraj.project['ConfFilename'], metric)

    traj = vtraj.load(CONF)
    
    ptraj = METRIC.prepare_trajectory(traj)
    
    n_frames = len(traj)
    
    distances = np.zeros(n_frames)
    assignments = np.zeros(n_frames, dtype=int)
    
    for i in xrange(n_frames):
        d_o2a = METRIC.one_to_all(ptraj, PGENS, i)
        #d = np.zeros(len(ptraj))
        assignments[i] = np.argmin(d_o2a)
        distances[i] = d_o2a[assignments[i]]
        
        
    return assignments, distances, vtraj
