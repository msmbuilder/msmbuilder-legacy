import sys
from abc import abstractmethod
from mpi4py import MPI as _MPI
import time
import numpy as np
from Queue import Queue
import threading
_COMM = _MPI.COMM_WORLD
_RANK = _COMM.rank
_SIZE = _COMM.size

def _kill_workers():
    for i in xrange(1, _SIZE):
        _COMM.isend(Job(method='die').msg, dest=i, tag=0)


class _Reciever(threading.Thread):
    def __init__(self, inbox, worker_list):
        threading.Thread.__init__(self)
        self.daemon = True
        self.inbox = inbox
        self.worker_list = worker_list
        self.start()
        
    def run(self):
        status = _MPI.Status()
        while True:
            if _COMM.Iprobe(source=_MPI.ANY_SOURCE, tag=1, status=status):
                i = status.Get_source()
                val, id, mailbox = _COMM.recv(source=i, tag=_MPI.ANY_TAG)

                self.worker_list[i] = True
                mb = getattr(self, mailbox)
                mb.put((val, id))
            else:
                time.sleep(0.05)
        

class _Sender(threading.Thread):
    def __init__(self, outbox, worker_list):
        threading.Thread.__init__(self)
        self.daemon = True
        self.outbox = outbox
        self.worker_list = worker_list
        self.start()
        
    def run(self):
        while True:
            job = self.outbox.get()
            target = job.target
            
            if target is None:
                target = self.worker_list.first_true()
            else:
                if not 1 <= target < _SIZE:
                    raise ValueError(('{target} must be greater than '
                                      'than 0 and less than {size}'.format(
                                target=target, size=_SIZE)))
        
            #print 'sending t={target}, m={msg}'.format(target=target, msg=job.msg)
            self.worker_list[target] = False
            _COMM.isend(job.msg, dest=target, tag=0)

class _AvailableList(object):
    def __init__(self, size):
        self._list = [True] * size
        self._list[0] = None
        self.lock = threading.Lock()
        self.on_set = threading.Event()

    def acquire(self):
        self.lock.acquire()

    def release(self):
        self.lock.release()

    def __getitem__(self, i):
        self.lock.acquire()
        val = self._list[i]
        self.lock.release()
        return val

    def __setitem__(self, key, val):
        self.lock.acquire()
        self._list[key] = val
        self.on_set.set()
        self.lock.release()
    
    def _first_true(self):
        self.lock.acquire()
        val = None
        for i, elem in enumerate(self._list):
            if elem:
                val = i
                break
        if val is None:
            self.on_set.clear()
        self.lock.release()
        return val

    def first_true(self):
        val = self._first_true()
        while val is None:
            self.on_set.wait()
            val = self._first_true()
        return val

class Job(object):
    def __init__(self, method, target=None, args=None, kwargs=None,
                 id=0, mailbox='inbox'):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        
        self.method = method
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.id = int(id)
        self.mailbox = mailbox
    
    @property
    def msg(self):
        return (self.method, self.args, self.kwargs, self.id, self.mailbox)

    def __repr__(self):
        return str((self.target, self.msg))


class BaseMaster(object):
    def __init__(self, **kwargs):
        self.inbox = Queue()
        self.outbox = Queue()
        self.worker_list = _AvailableList(_SIZE)
        self._reciever = _Reciever(self.inbox, self.worker_list)
        self._sender = _Sender(self.outbox, self.worker_list)
        self.__dict__.update(kwargs)
        
        self.run()
        
    def run(self):
        raise NotImplementedError('You must implement your own run method')

    def enqueue(self, method, *args, **kwargs):
        self.outbox.put(Job(method=method, args=args,
                            kwargs=kwargs))
    
    @property
    def workers(self):
        return xrange(1, _SIZE)

    def send(self, target, method, *args, **kwargs):
        self.outbox.put(Job(target=target, method=method,
                            args=args, kwargs=kwargs))

    def log(self, msg):
        print 'Master: {time}: {msg}'.format(time=time.strftime('%m/%d/%y [%l:%M:%S]'), msg=msg)
        
    def map(self, method, *args, **kwargs):
        length = None

        mailbox = 'box_%d' % np.random.randint(sys.maxint)
        setattr(self._reciever, mailbox, Queue())
        q = getattr(self._reciever, mailbox)

        for e in args:
            if length is None: length = len(e)
            assert len(e) == length, 'same length!'
        for k in kwargs.keys():
            if length is None: length = len(kwargs[k])
            assert len(kwargs[k]) == length, 'same length!'
        
        def job(i):
            jargs = tuple([args[k][i] for k in range(len(args))])
            jkwargs = {}
            for k in kwargs.keys():
                jkwargs[k] = kwargs[k][i]
            id = np.random.randint(sys.maxint)
            return Job(method=method, args=jargs, kwargs=jkwargs,
                       id=id, mailbox=mailbox)
        
        jobs = [job(i) for i in range(length)]
        ids = np.array([job.id for job in jobs])
        results = [None for job in jobs]
        num_returned = 0
        
        for j in jobs:
            self.outbox.put(j)
        
        while num_returned < length:
            val, id = q.get()
            i = np.where(id == ids)[0][0]
            results[i] = val
            num_returned += 1
        
        delattr(self._reciever, mailbox)
        return results

    
class BaseWorker(object):
    def _recv(self):
        method, args, kwargs, id, mailbox = _COMM.recv(source=0)
        if method == 'die':
            _COMM.send((None, None, None), tag=1)
            sys.exit(0)
        
        try:
            func = getattr(self, method)
            val = func(*args, **kwargs)
        except Exception as e:
            val = e

        _COMM.isend((val, id, mailbox), tag=1)

    def __init__(self, **kwargs):
        self.rank = _RANK
        self.__dict__.update(kwargs)
        while True:
            self._recv()

    def log(self, msg):
        msg = str(msg).strip()
        if len(msg) > 0:
            print 'Worker {r}/{s}: {time}: {msg}'.format(time=time.strftime('%m/%d/%y [%l:%M:%S]'), msg=msg,
                r = self.rank, s=_SIZE)
                

def start(master, worker, master_kwargs=None, worker_kwargs=None):
    if master_kwargs is None:
        master_kwargs = {}
    if worker_kwargs is None:
        worker_kwargs = {}

    if _SIZE < 2:
        raise RuntimeError('Execute me with mpirun and more than 1 process!')
    if not issubclass(master, BaseMaster):
        raise TypeError('arg1 must be a subclass of BaseMaster')
    if not issubclass(worker, BaseWorker):
        raise TypeError('arg2 must be a subclass of BaseWorker')
    
    if _RANK == 0:
        try:
            m = master(**master_kwargs)
        except:
            _kill_workers()
            raise
        finally:
            _kill_workers()

    else:
        w = worker(**worker_kwargs)
        

if __name__ == '__main__':
    class Master1(BaseMaster):
        def run(self):
            for w in self.workers:
                self.send(w, 'setup', 10)                
            
            print self.map('m', [1,2,3,4,5])
            

    class Worker1(BaseWorker):
        def setup(self, val):
            print 'setup on rank={rank}'.format(rank=self.rank)
            self.val = val
            time.sleep(2)
                    
        def m(self, i):
            print 'm on rank={rank}'.format(rank=self.rank)
            return i**2  * self.val
            
    start(Master1, Worker1)

    
