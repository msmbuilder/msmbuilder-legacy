import numpy as np
import pandas as pd
import scipy.sparse

def check_assignment_DataFrame_input(assignments):
    pass

def check_assignment_Series_input(assignments):
    pass

def convert_array_to_pandas(assignments):
    data = []
    num_traj, traj_length = assignments.shape
    for i in range(num_traj):
        for j, state in enumerate(assignments[i]):
            if state != -1:
                data.append([i, j, state])
    assignments = pd.DataFrame(data, columns=["traj", "time", "state"])
    return assignments


def get_count_matrix_from_assignments(assignments, n_states=None, lag_time=1, sliding_window=True):
    """
    Calculate counts matrix from `assignments`.

    Parameters
    ----------
    assignments : ndarray
        A 2d ndarray containing the state assignments.
    n_states : int, optional
        Can be automatically determined, unless you want a model with more states than are observed
    lag_time: int, optional
        the LagTime with which to estimate the count matrix. Default: 1
    sliding_window: bool, optional
        Use a sliding window.  Default: True

    Returns
    -------
    counts : sparse matrix
        `Counts[i,j]` stores the number of times in the assignments that a
        trajectory went from state i to state j in `LagTime` frames

    Notes
    -----
    assignments are input as iterables over numpy 1-d arrays of integers.
    For example a 2-d array where assignments[i,j] gives the ith trajectory, jth frame.
    The beginning and end of each trajectory may be padded with negative ones, which will be ignored.
    If the number of states is not given explitly, it will be determined as one plus the largest state index of the Assignments.
    Sliding window yields non-independent samples, but wastes less data.
    """

    check_assignment_DataFrame_input(assignments)

    if not n_states:
        n_states = assignments["state"].max() + 1

    C = scipy.sparse.lil_matrix((int(n_states), int(n_states)), dtype='float32')  # Lutz: why are we using float for count matrices?

    for (k, ass_i) in assignments.groupby("traj"):
        ass_i = ass_i.pivot_table(rows=["time"])["state"]
        C = C + get_counts_from_traj(ass_i, n_states, lag_time=lag_time, sliding_window=sliding_window)  # .tolil()

    return C


def get_counts_from_traj(states, n_states=None, lag_time=1, sliding_window=True):
    """Computes the transition count matrix for a sequence of states (single trajectory).

    Parameters
    ----------
    states : array
        A one-dimensional array of integers representing the sequence of states.
        These integers must be in the range [0, n_states]
    n_states : int
        The total number of states. If not specified, the largest integer in the
        states array plus one will be used.
    lag_time : int, optional
        The time delay over which transitions are counted
    sliding_window : bool, optional
        Use sliding window

    Returns
    -------
    C : sparse matrix of integers
        The computed transition count matrix
    """

    check_assignment_Series_input(states)

    if not n_states:
        n_states = states.max() + 1

    if sliding_window:
        from_states = states
        to_states = states.shift(-lag_time)
    else:
        states = states[::lag_time]
        from_states = states
        to_states = states.shift(-1)

    assert from_states.shape == to_states.shape

    transitions = pd.DataFrame([from_states, to_states]).dropna(axis=1)
    counts = np.ones(transitions.shape[1], dtype=int)

    transitions = transitions.values  # convert DataFrame to array
    C = scipy.sparse.coo_matrix((counts, transitions), shape=(n_states, n_states))

    return C
