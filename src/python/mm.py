import Queue
from collections import deque
import time
from mpi4py import MPI
import threading


comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
node = MPI.Get_processor_name()

# tags
INIT = 1
JOB_MSG = 2
RESULT = 3
ABORT = 4
MAP_DONE = 5
__initialization_args = None

class JobQueueEmpty(Exception):
    def __init__(self, outstanding_procs):
        self.outstanding_procs = outstanding_procs
        
class AsyncResults(object):
    def __init__(self, map_results_q, max_size):
        self._map_results_q = map_results_q
        self._max_size = max_size
        self._results = [None] * self._max_size
        self._slots_filled = 0
        
    def finished(self):
        '''Have all the results been returned (by the last call to get
        or get_nowait)'''
        return self._slots_filled == self._max_size
        
    def get(self, block=True, timeout=None):
        if block and timeout is None:
            while self._slots_filled < self._max_size:
                # item should be a 2 elem tuple. 1st is index,
                # second is result
                item = self._map_results_q.get(block=True)
                self._results[item[0]] = item[1]
                self._slots_filled += 1
        elif block and timeout is not None:
            start_time = time.time()
            while self._slots_filled < self._max_size:
                used_up = time.time() - start_time
                remaining = max(timeout - used_up, 0)
                try:
                    item = self._map_results_q.get(block=True, timeout=remaining)
                    self._results[item[0]] = item[1]
                    self._slots_filled += 1
                except Queue.Empty:
                    break
        elif not block:
            while self._slots_filled < self._max_size:
                try:
                    item = self._map_results_q.get_nowait()
                    self._results[item[0]] = item[1]
                    self._slots_filled += 1
                except Queue.Empty:
                    break
                    
        return self._results



def map(method, iterable):
    length = len(iterable)
    map_results_q = Queue.Queue()
    root_job_q = Queue.Queue()
    root_results_q = Queue.Queue()
    threading.Thread(target=_map_control, args=(method, iterable, map_results_q, root_job_q, root_results_q)).start()
    
    _root_worker(method, root_job_q, root_results_q)
    
    async = AsyncResults(map_results_q, length)
    results = async.get() # this blocks
    
    return results
    


def initialize_workers(args):
    global __initialization_args
    for i in range(1, size):
        comm.send(msg, dest=i, tag=INIT)
        
    # this global is for communicating to the root worker
    __initialization_args = args


def _map_control(method, iterable, map_results_q, root_job_q, root_results_q, root_worker=True):
    '''Job control'''
    
    iterable = deque(iterable)
    try:
        # This is a counter that gets dispatched with
        # each job so that the results can be put back in the right place
        # in the results array
        job_id = 0
        outstanding_procs = [False] * size # Which procs are runnning?
        
        # Send the first job to each process
        if root_worker:
            start = 0
        else:
            start = 1
            
        for i in xrange(start, size):
            if len(iterable) == 0:
                raise JobQueueEmpty(outstanding_procs)
            if i == 0:
                root_job_q.put((job_id, method, JOB_MSG, iterable.pop()))
            else:
                comm.send((job_id, method, JOB_MSG, iterable.pop()), dest=i, tag=JOB_MSG)
            job_id += 1
            outstanding_procs[i] = True
            
        while True:
            for i in xrange(0, size):
                if i == 0:
                    if root_worker:
                        try:
                            recvd_id, msg = root_results_q.get_nowait()
                            map_results_q.put_nowait((recvd_id, msg)) # return resultts
                            if len(iterable) == 0:
                                outstanding_procs[0] = False
                                raise JobQueueEmpty(outstanding_procs)
                            root_job_q.put((job_id, method, JOB_MSG, iterable.pop()))
                            job_id += 1
                        except Queue.Empty:
                            pass
                elif comm.Iprobe(source=i, tag=JOB_DONE):
                    # Remove the results message from the queue
                    # and add it to the results array
                    recvd_id, msg = comm.recv(source=i, tag=JOB_DONE)
                    map_results_q.put_nowait((recvd_id, msg))
                    
                    if len(iterable) == 0:
                        outstanding_procs[i] = False
                        raise JobQueueEmpty(outstanding_procs)
                        
                    # send this worker the next job from the queue                        
                    comm.send((job_id, method, JOB_MSG, iterable.pop()), dest=i, tag=JOB_MSG)
                    job_id += 1
                    
    except JobQueueEmpty as e:
        # Now that we've emptied the Queue, wait on all the workers
        outstanding_procs = e.outstanding_procs
        
        if root_worker:
            if outstanding_procs[0]:
                recvd_id, msg = root_results_q.get()
                map_results_q.put_nowait((recvd_id, msg))
                outstanding_procs[0] = False
                
        while True:
            for i in xrange(1, size):
                if comm.Iprobe(source=i, tag=JOB_DONE):
                    recvd_id, msg = comm.recv(source=i, tag=JOB_DONE)
                    map_results_q.put_nowait((recvd_id, msg))
                    outstanding_procs[i] = False
            if outstanding_procs == [False] * size:
                break
                
        # Send the map_done command to the root worker
        if root_worker:
            root_job_q.put(None, None, MAP_DONE, None)

def _map_worker(worker_cls):
    while True:
        if comm.Iprobe(source=0, tag=INIT):
            args = comm.recv(source=0, tag=INIT)
            worker = worker_cls(*args)

        if comm.Iprobe(source=0, tag=JOB_MSG):
            job_id, method, args = comm.recv(source=0, tag=INIT)
            result = getattr(worker, method)(*args)
            comm.send((job_id, result), dest=0, tag=JOB_DONE)
            
        if comm.Iprobe(source=0, tag=ABORT):
            sys.exit(1)

        time.sleep(0.1)


def _root_worker(worker_cls, job_q, results_q):
    global __initialization_args, __worker_cls
    print 'root worker seeing init_args', __initialization_args
    worker = __worker_cls(*__initialization_args)
    while True:
        j = job_q.get()
        job_id, method, tag, args = j
        
        if tag == MAP_DONE:
            break
        else:
            result = getattr(worker, method)(*args)
            results_q.put((job_id, result))

        time.sleep(0.1)

def kill_workers():
    # send all the abort signals to each of the workers
    for i in xrange(1, size):
        comm.send(0, dest=i, tag=ABORT)

def start(main, worker_cls):
    if rank == 0:
        global __worker_cls
        __worker_cls = worker_cls
        try:
            main()
        except:
            kill_workers()
            raise
        finally:
            kill_workers()
    else:
        _map_worker(worker_cls)