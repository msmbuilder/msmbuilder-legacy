import numpy as np
import scipy.sparse
import msmbuilder as msmb

n = 6

x = np.zeros((n, n))
np.fill_diagonal(x, 2.0)
x[0, 1] = 1.
x[1, 0] = 1
x[1, 2] = 2
x[2, 1] = 2

x[3, 4] = 1  # Second connected component
x[4, 3] = 1
x[2, 3] = 1  # WEAKLY connected to component 1


# Third connected component
x[5, 2] = 1  # WEAKLY connected to component 1

g = scipy.sparse.csr_matrix(x)
