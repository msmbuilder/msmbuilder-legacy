import numpy as np
from Emsmbuilder import _rg_wrap

def Reference_Rg(xyzlist):
    '''Return the radius of gyration of every frame in a xyzlist'''
    
    traj_length = len(xyzlist)
    Rg = np.zeros(traj_length)
    for i in xrange(traj_length):
        XYZ = xyzlist[i,:,:]
        mu = XYZ.mean(0)
        XYZ2 = XYZ-np.tile(mu,(len(XYZ),1))
        Rg[i] = (XYZ2**2.).mean()**0.5
        
        #print "xyz2", (XYZ2**2.0).mean()
        #print "frame", i
        #print "mean", mu
    
    return Rg
    
    
def Rg(xyzlist):
    '''
    Compute the Rg for every frame in an xyzlist
    
    If masses are none, then all the atoms are counted equally.
    Otherwise, you can supply a vector of length num_atoms giving
    the mass of each atom
    '''
    
    traj_length, num_atoms, num_dims = xyzlist.shape
    if not num_dims == 3:
        raise ValueError("What did you pass me?")
    if not xyzlist.dtype == np.float32:
        xyzlist = np.float32(xyzlist)
    
    results = np.zeros(traj_length, dtype=np.double)
    _rg_wrap.rg_wrap(xyzlist, results)
    return results
    
    
if __name__ == '__main__':
    from msmbuilder.Project import Project
    project = Project.LoadFromHDF('/home/rmcgibbo/Tutorial/ProjectInfo.h5')
    traj = project.LoadTraj(0)
    
    xyzlist = traj['XYZList']#[0:10,:,:]
    
    a = Reference_Rg(xyzlist)
    b = Rg(xyzlist)
    
    print a - b