import numpy as np
import scipy.weave
import warnings

# reference implementation
def __bond_angles(xyzlist, angle_indices):
    """Compute the bond angles for each frame in xyzlist
    
    This is a reference single threaded implementation in python/numpy
    
    Parameters
    ----------
    xyzlist : np.ndarray, shape=[n_frames, n_atoms, 3], dtype=float
        The cartesian coordinates
    angle_indices : np.ndarray, shape[n_angles, 3], dtype=int
        Each row gives the indices of three atoms which together make an angle
        
    Returns
    -------
    angles : np.ndarray, shape=[n_frames, n_angles], dtype=float
    """
    
    n_frames = xyzlist.shape[0]
    angles = np.zeros((n_frames, len(angle_indices)))

    for i in xrange(n_frames):
        for j, (m, o, n) in enumerate(angle_indices):
            u_prime = xyzlist[i, m, :] - xyzlist[i, o, :]
            v_prime = xyzlist[i, n, :] - xyzlist[i, o, :]
            u_norm = np.linalg.norm(u_prime)
            v_norm = np.linalg.norm(v_prime)

            angles[i, j] = np.arccos(np.dot(u_prime, v_prime) /
                (u_norm * v_norm))

    return angles

# multithreaded implementation
def bond_angles(xyzlist, angle_indices):
    """Compute the bond angles for each frame in xyzlist
    
    This is a OpenMP multithreaded parallel implementation in C++ using weave
    
    Parameters
    ----------
    xyzlist : np.ndarray, shape=[n_frames, n_atoms, 3], dtype=float
        The cartesian coordinates
    angle_indices : np.ndarray, shape[n_angles, 3], dtype=int
        Each row gives the indices of three atoms which together make an angle
        
    Returns
    -------
    angles : np.ndarray, shape=[n_frames, n_angles], dtype=float
    """
    
    # check shapes
    n_frames, n_atoms, n_dims = xyzlist.shape
    if not n_dims == 3:
        raise ValueError("xyzlist must be an n x m x 3 array")
    try: 
        n_angles, width = angle_indices.shape
        assert width is 3
    except (AttributeError, ValueError, AssertionError):
        raise ValueError('angle_indices must be an n x 3 array')
    
    # check type
    if xyzlist.dtype != np.float32:
        warnings.warn("xyzlist is not float32: copying", RuntimeWarning)
        xyzlist = np.array(xyzlist, dtype=np.float32)
    if angle_indices.dtype != np.int32:
        warnings.warn("angle_indpyices is not int32: copying", RuntimeWarning)
        angle_indices = np.array(angle_indices, dtype=np.int32)
    
    # make sure contiguous
    if not xyzlist.flags.c_contiguous:
        warnings.warn("xyzlist is not contiguous: copying", RuntimeWarning)
        xyzlist = np.copy(xyzlist)
    if not angle_indices.flags.c_contiguous:
        warnings.warn("angle_indices is not contiguous: copying", RuntimeWarning)
        angle_indices = np.copy(angle_indices)
    
    angles = np.zeros((n_frames, len(angle_indices)), dtype=np.double)
    
    scipy.weave.inline(r"""
    Py_BEGIN_ALLOW_THREADS
    int i, j, m, o, n;
    double up_x, up_y, up_z;
    double vp_x, vp_y, vp_z;
    double norm_u, norm_v;
    
    #pragma omp parallel for private(j, m, o, n, up_x, up_y, up_z, vp_x, vp_y, vp_z, norm_u, norm_v) shared(n_frames, n_angles, n_atoms, angles, xyzlist)
    for (i = 0; i < n_frames; i++) {
        for (j = 0; j < n_angles; j++) {
            m = angle_indices[j*3 + 0];
            n = angle_indices[j*3 + 1];
            o = angle_indices[j*3 + 2];
            
            up_x = xyzlist[i*n_atoms*3 + m*3 + 0] - xyzlist[i*n_atoms*3 + n*3 + 0];
            up_y = xyzlist[i*n_atoms*3 + m*3 + 1] - xyzlist[i*n_atoms*3 + n*3 + 1];
            up_z = xyzlist[i*n_atoms*3 + m*3 + 2] - xyzlist[i*n_atoms*3 + n*3 + 2];

            vp_x = xyzlist[i*n_atoms*3 + o*3 + 0] - xyzlist[i*n_atoms*3 + n*3 + 0];
            vp_y = xyzlist[i*n_atoms*3 + o*3 + 1] - xyzlist[i*n_atoms*3 + n*3 + 1];
            vp_z = xyzlist[i*n_atoms*3 + o*3 + 2] - xyzlist[i*n_atoms*3 + n*3 + 2];
            
            norm_u = sqrt(up_x*up_x + up_y*up_y + up_z*up_z);
            norm_v = sqrt(vp_x*vp_x + vp_y*vp_y + vp_z*vp_z);
            
            angles[i*n_angles + j] = acos((up_x*vp_x + up_y*vp_y + up_z*vp_z) / (norm_u * norm_v));
        }
    }
    Py_END_ALLOW_THREADS
    """, ["xyzlist", "angle_indices", "angles", "n_frames",
          "n_angles", "n_atoms"],
         extra_compile_args = ["-O3", "-fopenmp"],  
         extra_link_args=['-lgomp'],
         compiler='gcc')
    # note that weave by default includes math.h in the generated cpp file, which
    # declares sqrt and acos

    return angles

