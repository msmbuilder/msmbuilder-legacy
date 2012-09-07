from hashlib import sha1
import numpy
import tables
from msmbuilder import Trajectory
from msmbuilder.Trajectory import _ConvertFromLossyIntegers

class Chunk(object):
    def __init__(self, traj, start, stop):
        self.traj = traj
        self.start = start
        self.stop = stop

    def __repr__(self):
        return str((self.traj, self.start, self.stop))

    def __iter__(self):
        return (e for e in (self.traj, self.start, self.stop))

    def __len__(self):
        return self.stop - self.start
        
class VTraj(object):
    """A "Virtual Trajectory" -- a logical collection of frames that do not
    necessarily come from the same physical trajectory.
    
    Each VTraj is represented as a list of Chunks. Each Chunk represents a slice
    of a physical trajectory, and itself is represented by the index of the
    physical trajectory, and a starting ending index (of where) the slice is
    
    The VTraj provides a load() method to load the frames from disk.
    
    """

    def __init__(self, project, *args):
        self.chunks = []
        self.project = project
        
        for arg in args:
            if isinstance(arg, tuple):
                self.append(Chunk(*arg))
            elif isinstance(arg, Chunk):
                self.append(arg)
            else:
                raise TypeError()
            
    def __repr__(self):
        return str(self.chunks)
    
    def append(self, arg):
        if isinstance(arg, tuple):
            self.chunks.append(Chunk(*arg))
        elif isinstance(arg, Chunk):
            self.chunks.append(arg)
        else:
            raise TypeError()
    
    def __iter__(self):
        return (e for e in self.chunks)
    
    def __len__(self):
        return sum(len(e) for e in self.chunks)
        
    def hash(self):
        """Unique identifier of VTraj
        
        The identifier is based only on the chunks, and not on the actual data
        in the project file.
        
        Returns
        -------
        hash : str
            A unique identifier (sha1 hash)
        """
        return sha1(str([str(e) for e in self.chunks])).hexdigest()
        
    def load(self, conf):
        """Load the coordinates and get a physical trajectory
        
        The filename to load from is taken from the project file.
        
        Parameters
        ----------
        conf : msmbuilder.Trajectory
            When we load the trajectories from disk, at this point we're only
            getting the XYZ coordinates. The XYZ coordinates will be injected
            into conf as a "container"
            
        Returns
        -------
        traj : msmbuilder.Trajectory
            This is `conf`, with the XYZ coordinates loaded from disk
        """
        n_atoms = conf.GetNumberOfAtoms()
        
        xyzlist = numpy.zeros((len(self), n_atoms, 3), dtype=numpy.float32)
        last_frame = 0
        
        for trj_i, start, stop in self.chunks:
            f = tables.File(self.project.GetTrajFilename(trj_i))
            frames = _ConvertFromLossyIntegers(f.root.XYZList[start:stop], 1000)
            
            xyzlist[last_frame:last_frame + len(frames), :, :] = frames
            last_frame += len(frames)
            
            f.close()
            
        conf['XYZList'] = xyzlist
        
        return conf
        
    def canonical(self):
        """Simple representation for testing
        
        Returns
        -------
        canonical_rep : list
            representation of the VTraj as a list of (traj, start, stop)
            tuples.
        """
        
        return list(tuple(e) for e in self)
