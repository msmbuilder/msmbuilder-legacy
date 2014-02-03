import numpy as np
import mdtraj as md

def points_on_cube():
    """Generate example data.
    
    Returns
    -------
    
    traj : mdtraj.Trajectory
        Trajectory containing points on cube.
    
    Notes
    -----
    
    This generates an mdtraj trajectory with 8 frames and 4 atoms.  Of the
    8 frames, there are only 4 distinct conformations.  In particular
    traj.xyz[i] == traj.xyz[i + 4] for all i.
    
    If we cluster this data into 4 clusters, we expect there to be 2 
    conformations assigned to each cluster.  We also expect the distance
    to each generator to be zero.  
    
    """

    n_frames = 8
    n_atoms = 4
    top = md.Topology()
    chain = top.add_chain()
    residue = top.add_residue("UNK", chain)
    
    pose0 = np.zeros((n_atoms, 3))
    
    pose1 = np.zeros((n_atoms, 3))
    pose1[3, 0] = 1
    
    pose2 = np.zeros((n_atoms, 3))
    pose2[2, 1] = 1
    pose2[3, 1] = 2
    
    pose3 = np.zeros((n_atoms, 3))
    
    pose0[0, 2] = 0.
    pose0[1, 2] = 1.
    pose0[2, 2] = 2.
    pose0[3, 2] = 3.
    
    
    for i in range(n_atoms):
        atom = top.add_atom("H%d" % i, None, residue)
    
    xyz = np.array([pose0, pose1, pose2, pose3, pose0, pose1, pose2, pose3])
    traj = md.Trajectory(xyz, top)
    return traj

if __name__ == "__main__":
    traj = points_on_cube()
    traj.save("./reference/points_on_cube/Trajectories/trj0.lh5")
    traj = traj[0:1]
    traj.save("./reference/points_on_cube/native.pdb")
    atom_indices = np.arange(traj.n_atoms)
    np.savetxt("./reference/points_on_cube/AtomIndices.dat", atom_indices, "%d")
