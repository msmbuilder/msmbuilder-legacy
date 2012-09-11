import os, sys
import inspect
import numpy as np
from msmbuilder import Trajectory
from gpurmsd.gpurmsd import GPURMSD
from msmbuilder.metrics import RMSD
import matplotlib.pyplot as pp
import numpy.testing as npt

def fixtures_dir():
    #http://stackoverflow.com/questions/50499/in-python-how-do-i-get-the-path-and-name-of-the-file-that-is-currently-executin
    return os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), 'fixtures')

trj_path = os.path.join(fixtures_dir(), 'trj0.lh5')
ww_conf = Trajectory.LoadTrajectoryFile(os.path.join(fixtures_dir(), 'ww.pdb'))
ww_1 = os.path.join(fixtures_dir(), 'ww.xtc')
ww_2 = os.path.join(fixtures_dir(), 'ww-aligned.xtc')

def test_gpurmsd():
    traj = Trajectory.LoadTrajectoryFile(trj_path)    

    gpurmsd = GPURMSD()
    ptraj = gpurmsd.prepare_trajectory(traj)
    gpurmsd._gpurmsd.print_params()
    gpu_distances = gpurmsd.one_to_all(ptraj, ptraj, 0)

    cpurmsd = RMSD()
    ptraj = cpurmsd.prepare_trajectory(traj)
    cpu_distances = cpurmsd.one_to_all(ptraj, ptraj, 0)
    
    npt.assert_array_almost_equal(cpu_distances, gpu_distances, decimal=4)

def plot_gpu_cmd_correlation():
    traj1 = Trajectory.LoadTrajectoryFile(ww_1, Conf=ww_conf)
    traj1_copy = Trajectory.LoadTrajectoryFile(ww_1, Conf=ww_conf)
    #traj2 = Trajectory.LoadTrajectoryFile(ww_2, Conf=ww_conf)
    #traj2_copy = Trajectory.LoadTrajectoryFile(ww_2, Conf=ww_conf)

    def gpudist(t):
        gpurmsd = GPURMSD()
        pt = gpurmsd.prepare_trajectory(t)
        gpurmsd._gpurmsd.print_params()
        return gpurmsd.one_to_all(pt, pt, 0)
    def cpudist(t):
        rmsd = RMSD()
        pt = rmsd.prepare_trajectory(t)
        return rmsd.one_to_all(pt, pt, 0)
    g1 = gpudist(traj1) #, gpudist(traj2)
    c1 = cpudist(traj1_copy) #, cpudist(traj2_copy)

    pp.subplot(231)
    pp.plot(c1)
    pp.title('cpu rmsd drift along traj')
    pp.xlabel('frame index')
    pp.xlabel('cpurmsd($X_{0}$, $X_{frame_index}$)')

    pp.subplot(232)
    pp.scatter(g1, c1)
    pp.xlabel('gpu rmsd')
    pp.ylabel('cpu rmsd')

    pp.subplot(233)
    pp.plot(g1)
    pp.title('gpu rmsd drift along traj')
    pp.xlabel('frame index')
    pp.xlabel('gpurmsd($X_{0}$, $X_{frame_index}$)')


    #PLOT c2 and g2 in the lower portion of the graph

    #pp.subplot(234)
    #pp.plot(c2)
    #pp.title('cpu rmsd drift along pre-aligned traj')
    #pp.xlabel('frame index')
    #pp.xlabel('cpurmsd($X_{0}$, $X_{frame_index}$)')

    #pp.subplot(235)
    #pp.scatter(g2, c2)
    #pp.xlabel('gpu rmsd')
    #pp.ylabel('cpu rmsd')

    #pp.subplot(236)
    #pp.plot(g2)
    #pp.title('gpu rmsd drift along pre-aligned traj')
    #pp.xlabel('frame index')
    #pp.xlabel('gpurmsd($X_{0}$, $X_{frame_index}$)')

    #pp.subplots_adjust(hspace=0.4)
    #pp.savefig('gpucpu_correlation.png')
    pp.show()
    

if __name__ == '__main__':
    plot_gpu_cmd_correlation()
