import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
import scipy.stats


def GetFormat():
    """Construct a generator to enumerate colors for plotting.
    
    Returns
    -------
    ColorList[i] : string
        Color string

    """
    ColorList = [
        "#99CC00",
        "#003300",
        "#00CC99",
        "#0000FF",
        "#6600FF",
        "#CC0000",
        "#FF9900",
        "#FFFF99",
        "#000000",
        "#CCCCCC"
        ]
    i = 0
    while True:
        i = (i + 1) % len(ColorList)
        yield ColorList[i]

color_generator = GetFormat()


def FixEntry(mapping, parameters, populations, K0, i, j, Val):
    """Constrain an entry in a rate matrix.
    
    Parameters
    ----------
    mapping : list
        List of (i,j) indices that have already been constrained.
    parameters : list
        List of rate matrix elements K[i,j].
    populations : ndarray
        Equilibrium populations
    i : int
    j : int
    Val : float
        Constrain the value K[i,j] = Val.
    
    Notes
    -----
    Also constrains the transpose entry by detailed balance.
    Removes the entry from the lists of free variables (mapping, parameters).
    
    """
    
    Ind = mapping.index([i, j])
    mapping.pop(Ind)
    parameters.pop(Ind)
    K0[i, j] = Val
    K0[j, i] = Val * populations[i] / populations[j]


def LogLikelihood(C, T):
    """Calculate the likelihood of a transition matrix given counts C.

    Parameters
    ----------
    C : ndarray
        Count matrix
    T : ndarray
        Transition matrix
    
    Returns
    ----------
    f : float
        log likelihood of T given C
    
    """
    
    f = np.sum(C * np.log(T))
    return f


def ConvertTIntoK(T0):
    """Convert a transition matrix into a rate matrix.

    Parameters
    ----------
    T0 : ndarray
        Initial transition matrix
    
    Returns
    ----------
    K : ndarray
        rate matrix

    
    Notes
    ----------
    For best results, use a transition matrix with short lagtime.
    This follows because one can show that
    T = exp(K t) ~ I +K t + ...
    
    """

    if type(T0) != np.ndarray:
        raise(Exception("Error, T0 must be a numpy array"))
    
    D = T0.diagonal()
    K = T0 - np.diag(D)
    D2 = K.sum(1)
    K = K - np.diag(D2)
    return(K)


def ConstructRateFromParams(parameters, mapping, populations, K0):
    """Construct a rate matrix from a flat array of parameters.

    Parameters
    ----------

    parameters : list
        Flat list of parameters
    mapping : list
        Mapping from flat indices to (2d) array indices.
    populations : ndarray
        stationary populations of model
    K0 : ndarray
        Initial rate matrix.
        
    Returns
    ----------
    K : ndarray
        rate matrix

    
    Notes
    ----------
    The returned matrix K is such that
     K[mapping.T[0],mapping.T[1]] = abs(parameters)

    """

    parameters = np.array(parameters)
    mapping = np.array(mapping)
    
    K = K0.copy()
    if len(mapping) > 0:
        K[mapping.T[0], mapping.T[1]] = abs(parameters)
        X2 = abs(parameters) * populations[mapping.T[0]] / populations[mapping.T[1]]
        K[mapping.T[1], mapping.T[0]] = X2

    K -= np.diag(K.sum(1))

    return K


def get_parameter_mapping(K):
    """Get a mapping from 1D to 2D indices.

    Parameters
    ----------

    K : ndarray
        Rate matrix.
        
    Returns
    ----------
    mapping : list
        List of nonzero (i,j) elements such that i > j.
    parameters : list
        List of values of K[i,j] at locations in mapping.
    
    """
    
    mapping = []
    parameters = []
    NumStates = K.shape[0]
    for i in range(NumStates):
        for j in range(NumStates):
            if i > j and K[i, j] != 0:
                mapping.append([i, j])
                parameters.append(K[i, j])

    return mapping, parameters
                

def MaximizeRateLikelihood(parameters, mapping, populations, C, K0):
    """Maximize the likelihood of a rate matrix given assignment data.

    Parameters
    ----------
    parameters : list
        List of values of K[i,j] at locations in mapping.
    mapping : list
        List of nonzero (i,j) elements such that i > j.
    C : ndarray
        Array of observed counts
    K0 : ndarray
        Initial rate matrix.
        
    Returns
    ----------
    ans : ndarray
        Final rate matrix.
    
    """

    def obj(parameters):
        K = ConstructRateFromParams(parameters, mapping, populations, K0)
        T = scipy.linalg.matfuncs.expm(K)
        f = LogLikelihood(C, T)
        return -1 * f

    def callback(parameters):
        pass

    ans = scipy.optimize.fmin(obj, parameters, full_output=True, xtol=1E-10, ftol=1E-10, maxfun=100000, maxiter=100000, callback=callback)[0]
    ans = abs(ans)
    return ans


def PlotRates(KList, LagTimeList, counts_list, Tau=1):
    """Plot the intermediate SCRE output to evaluate convergence.

    Parameters
    ----------
    KList : list
        List of MLE rate matrices as function of lagtime.
    LagTimeList : list
        List of lagtimes.
    counts_list : ndarray
        List of observed count arrays at function of lagtime.
    Tau : int, optional
        The time unit associated with spacings between lagtimes.
        
    """
    import matplotlib
    KList = np.array(KList)
    NumStates = KList.shape[-1]
    TauList = Tau / KList
    counts_list = np.array(counts_list)
    for i in range(NumStates):
        for j in range(NumStates):
            if i > j and KList[0, i, j] > 0:
                matplotlib.pyplot.errorbar(Tau * LagTimeList, TauList[:, i, j], fmt=color_generator.next(), yerr=TauList[:, i, j] / np.sqrt(counts_list[:, i]), label="%d-%d" % (i, j))
                #matplotlib.pyplot.plot(Tau*LagTimeList,TauList[:,i,j],color_generator.next(),label="%d-%d"%(i,j))

    matplotlib.pyplot.yscale('log')
    matplotlib.pyplot.legend(loc=0)
