"""
To USE this code, from a client perspective, all you want to do is

>> from metrics import RMSD, Dihedral, Contact

Nothing else in this modules' namespace will be useful to you as a client.

and then for example

>> rmsdtraj1 = RMSD.prepare_trajectory(traj, atomindices)
>> RMSD.one_to_all(rmsdtraj1, rmsdtraj1, 0)
>> dihedraltraj1 = Dihedral.prepare_trajectory(traj)
>> Dihedral.one_to_all(dihedraltraj1, dihedraltraj1, 0)

this would compute the distances from frame 0 to all other frames under both
the rmsd metric and dihedral metric. There are a lot more options and ways you can
calcuate distances (euclidean distance vs. cityblock vs pnorm, etc etc.) and select
the frames you care about (one_to_all(), one_to_many(), many_to_many(), all_to_all(), etc).

NOTE: Because the code leverages inheritance, if you just casually browse the code
for Dihedral for example, you ARE NOT going to see all methods that the class
actually implements. I would browsing the docstrings in ipython.

=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=

To DEVELOP on top of this code, just implement some classes that inherit from
either AbstractDistanceMetric or Vectorized. In particular, if you inherit
from Vectorized, you get a fully functional DistanceMetric class almost for
free. All you have to do is define a prepare_trajectory() fuction.

=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=

This should be documented better somewhere, because it will cause cryptic
errors if you don't do it. Whatever data structure you return from
prepare_trajectory() needs to support slice sytax. If you return an array or
something, then this is no problem, but if you create your own object to hold
the data that prepare_trajectory() returns, you need to add a __getitem__(),
 __setitem__() and __len__() methods. See the RMSD.TheoData object for an example. Also, if you're not familiar with this side of python these docs 
(http://docs.python.org/reference/datamodel.html#emulating-container-types)
are pretty good. Only __getitem__, __setitem__ and __len__ are necessary.

#=#=#=#+#+#+#+#+#+#+#+#+#+#+#+#+#
"""


import abc
import re
from msmbuilder import _rmsdcalc # this is a forked version of the msmbuilder rmsdcalc
# with a new method that does the one_to_many efficiently inside the c code
# since python array slicing can be really slow. Big speedup using this
# for the drift calculation
import copy
import numpy as np
import itertools
import scipy.spatial.distance
import warnings
from collections import defaultdict, namedtuple
from numbers import Number
from msmbuilder import _distance_wrap
from msmbuilder.Serializer import Serializer
#from msmbuilder import drift
from msmbuilder.geometry import dihedral as _dihedralcalc
from msmbuilder.geometry import contact as _contactcalc
from msmbuilder.geometry import rg as _rgcalc
import logging
logger = logging.getLogger('metrics')
#######################################################
# Toggle to use faster (no typechecking, so unsafe too)
# version of scipy.spatial.distance.cdist which is 
# parallelized with OMP pragmas. This is used for most
# of the Vectorized methods
#######################################################
USE_FAST_CDIST = True
#USE_FAST_CDIST = False
#######################################################

def fast_cdist(XA, XB, metric='euclidean', p=2, V=None, VI=None):
    r"""
    Computes distance between each pair of the two collections of inputs.
    
    This is a direct copy of a function in scipy (scipy.spatial.distance.cdist)
    except that we do fewer typechecks and then call out to an OpenMP parallelized
    version of the implementation code (for multicore)
    
    ``XA`` is a :math:`m_A` by :math:`n` array while ``XB`` is a :math:`m_B` by
    :math:`n` array. A :math:`m_A` by :math:`m_B` array is
    returned. An exception is thrown if ``XA`` and ``XB`` do not have
    the same number of columns.

    A rectangular distance matrix ``Y`` is returned. For each :math:`i`
    and :math:`j`, the metric ``dist(u=XA[i], v=XB[j])`` is computed
    and stored in the :math:`ij` th entry.

    
    The following are common calling conventions:

    1. ``Y = cdist(XA, XB, 'euclidean')``

       Computes the distance between :math:`m` points using
       Euclidean distance (2-norm) as the distance metric between the
       points. The points are arranged as :math:`m`
       :math:`n`-dimensional row vectors in the matrix X.

    2. ``Y = cdist(XA, XB, 'minkowski', p)``

       Computes the distances using the Minkowski distance
       :math:`||u-v||_p` (:math:`p`-norm) where :math:`p \geq 1`.

    3. ``Y = cdist(XA, XB, 'cityblock')``

       Computes the city block or Manhattan distance between the
       points.

    4. ``Y = cdist(XA, XB, 'seuclidean', V=None)``

       Computes the standardized Euclidean distance. The standardized
       Euclideanan distance between two n-vectors ``u`` and ``v`` is

       .. math::

          \sqrt{\sum {(u_i-v_i)^2 / V[x_i]}}.

       V is the variance vector; V[i] is the variance computed over all
          the i'th components of the points. If not passed, it is
          automatically computed.

    5. ``Y = cdist(XA, XB, 'sqeuclidean')``

       Computes the squared Euclidean distance :math:`||u-v||_2^2` between
       the vectors.

    6. ``Y = cdist(XA, XB, 'cosine')``

       Computes the cosine distance between vectors u and v,

       .. math::

          1 - \frac{u \cdot v}
                   {{||u||}_2 {||v||}_2}

       where :math:`||*||_2` is the 2-norm of its argument ``*``, and
       :math:`u \cdot v` is the dot product of :math:`u` and :math:`v`.

    7. ``Y = cdist(XA, XB, 'correlation')``

       Computes the correlation distance between vectors u and v. This is

       .. math::

          1 - \frac{(u - \bar{u}) \cdot (v - \bar{v})}
                   {{||(u - \bar{u})||}_2 {||(v - \bar{v})||}_2}

       where :math:`\bar{v}` is the mean of the elements of vector v,
       and :math:`x \cdot y` is the dot product of :math:`x` and :math:`y`.


    8. ``Y = cdist(XA, XB, 'hamming')``

       Computes the normalized Hamming distance, or the proportion of
       those vector elements between two n-vectors ``u`` and ``v``
       which disagree. To save memory, the matrix ``X`` can be of type
       boolean.

    9. ``Y = cdist(XA, XB, 'jaccard')``

       Computes the Jaccard distance between the points. Given two
       vectors, ``u`` and ``v``, the Jaccard distance is the
       proportion of those elements ``u[i]`` and ``v[i]`` that
       disagree where at least one of them is non-zero.

    10. ``Y = cdist(XA, XB, 'chebyshev')``

       Computes the Chebyshev distance between the points. The
       Chebyshev distance between two n-vectors ``u`` and ``v`` is the
       maximum norm-1 distance between their respective elements. More
       precisely, the distance is given by

       .. math::

          d(u,v) = \max_i {|u_i-v_i|}.

    11. ``Y = cdist(XA, XB, 'canberra')``

       Computes the Canberra distance between the points. The
       Canberra distance between two points ``u`` and ``v`` is

       .. math::

         d(u,v) = \sum_i \frac{|u_i-v_i|}
                              {|u_i|+|v_i|}.

    12. ``Y = cdist(XA, XB, 'braycurtis')``

       Computes the Bray-Curtis distance between the points. The
       Bray-Curtis distance between two points ``u`` and ``v`` is


       .. math::

            d(u,v) = \sum{\sum_i (u_i-v_i)}
                          {\sum_i (u_i+v_i)}

    13. ``Y = cdist(XA, XB, 'mahalanobis', VI=None)``

       Computes the Mahalanobis distance between the points. The
       Mahalanobis distance between two points ``u`` and ``v`` is
       :math:`(u-v)(1/V)(u-v)^T` where :math:`(1/V)` (the ``VI``
       variable) is the inverse covariance. If ``VI`` is not None,
       ``VI`` will be used as the inverse covariance matrix.

    14. ``Y = cdist(XA, XB, 'yule')``

       Computes the Yule distance between the boolean
       vectors. (see yule function documentation)

    15. ``Y = cdist(XA, XB, 'matching')``

       Computes the matching distance between the boolean
       vectors. (see matching function documentation)

    16. ``Y = cdist(XA, XB, 'dice')``

       Computes the Dice distance between the boolean vectors. (see
       dice function documentation)

    17. ``Y = cdist(XA, XB, 'kulsinski')``

       Computes the Kulsinski distance between the boolean
       vectors. (see kulsinski function documentation)

    18. ``Y = cdist(XA, XB, 'rogerstanimoto')``

       Computes the Rogers-Tanimoto distance between the boolean
       vectors. (see rogerstanimoto function documentation)

    19. ``Y = cdist(XA, XB, 'russellrao')``

       Computes the Russell-Rao distance between the boolean
       vectors. (see russellrao function documentation)

    20. ``Y = cdist(XA, XB, 'sokalmichener')``

       Computes the Sokal-Michener distance between the boolean
       vectors. (see sokalmichener function documentation)

    21. ``Y = cdist(XA, XB, 'sokalsneath')``

       Computes the Sokal-Sneath distance between the vectors. (see
       sokalsneath function documentation)

    22. ``Y = cdist(XA, XB, f)``

       Computes the distance between all pairs of vectors in X
       using the user supplied 2-arity function f. For example,
       Euclidean distance between the vectors could be computed
       as follows::

         dm = cdist(XA, XB, lambda u, v: np.sqrt(((u-v)**2).sum()))

       Note that you should avoid passing a reference to one of
       the distance functions defined in this library. For example,::

         dm = cdist(XA, XB, sokalsneath)

       would calculate the pair-wise distances between the vectors in
       X using the Python function sokalsneath. This would result in
       sokalsneath being called :math:`{n \choose 2}` times, which
       is inefficient. Instead, the optimized C version is more
       efficient, and we call it using the following syntax.::

         dm = cdist(XA, XB, 'sokalsneath')

    Parameters
    ----------
    XA : ndarray
        An :math:`m_A` by :math:`n` array of :math:`m_A`
        original observations in an :math:`n`-dimensional space.
    XB : ndarray
        An :math:`m_B` by :math:`n` array of :math:`m_B`
        original observations in an :math:`n`-dimensional space.
    metric : string or function
        The distance metric to use. The distance function can
        be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
        'jaccard', 'kulsinski', 'mahalanobis', 'matching',
        'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.
    w : ndarray
        The weight vector (for weighted Minkowski).
    p : double
        The p-norm to apply (for Minkowski, weighted and unweighted)
    V : ndarray
        The variance vector (for standardized Euclidean).
    VI : ndarray
        The inverse of the covariance matrix (for Mahalanobis).
        
    
    Returns
    -------
    Y : ndarray
        A :math:`m_A` by :math:`m_B` distance matrix.

    """
    
    mA = XA.shape[0]
    mB = XB.shape[0]
    
    if mB != 1:
        raise Exception('Known buggy when mB!=1')
    
    if mB > mA:
        raise Exception('The parallelism is the other way. switch them around.')
    if not ((XA.dtype == np.double and XB.dtype == np.double) or (XA.dtype == np.bool and XB.dtype == np.bool)):
        raise TypeError('The vectors need to be of type np.double or np.bool.')
    if not (XA.flags.contiguous and XB.flags.contiguous):
        raise Exception('Prepared trajectories need to be contiguous.')

    if not XA.shape[1] == XB.shape[1]:
        raise Exception('shape[1] mismatch')
    
    dm = np.empty((mA, mB), dtype=np.double)
    n = XA.shape[1]
    #dm = np.zeros((mA, mB), dtype=np.double)
        
    if metric == 'euclidean':
        _distance_wrap.cdist_euclidean_wrap(XA, XB, dm)
    elif metric == 'cityblock':
        _distance_wrap.cdist_city_block_wrap(XA, XB, dm)
    elif metric == 'sqeuclidean':
        _distance_wrap.cdist_euclidean_wrap(XA, XB, dm)
        dm **= 2.0
    elif metric == 'hamming':
        if XA.dtype == np.bool:
            _distance_wrap.cdist_hamming_bool_wrap(XA, XB, dm)
        else:
            _distance_wrap.cdist_hamming_wrap(XA, XB, dm)
    elif metric == 'chebychev':
        _distance_wrap.cdist_chebyshev_wrap(XA, XB, dm)
    elif metric == 'minkowski':
        _distance_wrap.cdist_minkowski_wrap(XA, XB, dm, p)
    #elif metric == 'wminkowski':
    #    _distance_wrap.cdist_weighted_minkowski_wrap(XA, XB, dm)
    elif metric == 'seuclidean':
        if V is not None:
            V = np.asarray(V, order='c')
            if type(V) != np.ndarray:
                raise TypeError('Variance vector V must be a numpy array')
            if V.dtype != np.double:
                raise TypeError('Variance vector V must contain doubles.')
            if len(V.shape) != 1:
                raise ValueError('Variance vector V must be '
                                 'one-dimensional.')
            if V.shape[0] != n:
                raise ValueError('Variance vector V must be of the same '
                                 'dimension as the vectors on which the '
                                 'distances are computed.')
            if not V.flags.contiguous:
                raise ValueError('V must be contiguous')
            
            # The C code doesn't do striding.
            #[VV] = scipy.spatial.distance._copy_arrays_if_base_present([V])
        else:
            raise ValueError('You need to supply V')
        _distance_wrap.cdist_seuclidean_wrap(XA, XB, V, dm)
    elif metric == 'mahalanobis' or metric == 'sqmahalanobis':
        if VI is not None:
            VI = scipy.spatial.distance._convert_to_double(np.asarray(VI, order='c'))
            if type(VI) != np.ndarray:
                raise TypeError('VI must be a numpy array.')
            if VI.dtype != np.double:
                raise TypeError('The array must contain 64-bit floats.')
            [VI] = scipy.spatial.distance._copy_arrays_if_base_present([VI])
        else:
            raise ValueError('You must supply VI')
        _distance_wrap.cdist_mahalanobis_wrap(XA, XB, VI, dm)
        if metric == 'sqmahalanobis':
            dm **= 2.0        
    elif metric == 'cosine':
        normsA = np.sqrt(np.sum(XA * XA, axis=1))
        normsB = np.sqrt(np.sum(XB * XB, axis=1))
        _distance_wrap.cdist_cosine_wrap(XA, XB, dm, normsA, normsB)
    elif metric == 'braycurtis':
        _distance_wrap.cdist_bray_curtis_wrap(XA, XB, dm)
    elif metric == 'canberra':
        _distance_wrap.cdist_canberra_wrap(XA, XB, dm)
    elif metric == 'dice':
        _distance_wrap.cdist_dice_bool_wrap(XA, XB, dm)
    elif metric == 'kulsinki':
        _distance_wrap.cdist_kulsinski_bool_wrap(XA, XB, dm)
    elif metric == 'matching':
        _distance_wrap.cdist_matching_bool_wrap(XA, XB, dm)
    elif metric == 'rogerstanimoto':
        _distance_wrap.cdist_rogerstanimoto_bool_wrap(XA, XB, dm)
    elif metric == 'russellrao':
        _distance_wrap.cdist_russellrao_bool_wrap(XA, XB, dm)
    elif metric == 'sokalmichener':
        _distance_wrap.cdist_sokalmichener_bool_wrap(XA, XB, dm)
    elif metric == 'sokalsneath':
        _distance_wrap.cdist_sokalsneath_bool_wrap(XA, XB, dm)
    elif metric == 'yule':
        _distance_wrap.cdist_yule_bool_wrap(XA, XB, dm)
    elif metric == 'correlation':
        return scipy.spatial.distance.cdist(XA, XB, 'correlation')
    else:
        raise ValueError('Unknown Distance Metric: %s' % metric)
    
    return dm


def fast_pdist(X, metric='euclidean', p=2, V=None, VI=None):
    r"""
    Computes the pairwise distances between m original observations in
    n-dimensional space. Returns a condensed distance matrix Y.  For
    each :math:`i` and :math:`j` (where :math:`i<j<n`), the
    metric ``dist(u=X[i], v=X[j])`` is computed and stored in entry ``ij``.

    See ``squareform`` for information on how to calculate the index of
    this entry or to convert the condensed distance matrix to a
    redundant square matrix.

    The following are common calling conventions.

    1. ``Y = pdist(X, 'euclidean')``

       Computes the distance between m points using Euclidean distance
       (2-norm) as the distance metric between the points. The points
       are arranged as m n-dimensional row vectors in the matrix X.

    2. ``Y = pdist(X, 'minkowski', p)``

       Computes the distances using the Minkowski distance
       :math:`||u-v||_p` (p-norm) where :math:`p \geq 1`.

    3. ``Y = pdist(X, 'cityblock')``

       Computes the city block or Manhattan distance between the
       points.

    4. ``Y = pdist(X, 'seuclidean', V=None)``

       Computes the standardized Euclidean distance. The standardized
       Euclidean distance between two n-vectors ``u`` and ``v`` is

       .. math::

          \sqrt{\sum {(u_i-v_i)^2 / V[x_i]}}.


       V is the variance vector; V[i] is the variance computed over all
       the i'th components of the points.  If not passed, it is
       automatically computed.

    5. ``Y = pdist(X, 'sqeuclidean')``

       Computes the squared Euclidean distance :math:`||u-v||_2^2` between
       the vectors.

    6. ``Y = pdist(X, 'cosine')``

       Computes the cosine distance between vectors u and v,

       .. math::

          1 - \frac{u \cdot v}
                   {{||u||}_2 {||v||}_2}

       where :math:`||*||_2` is the 2-norm of its argument ``*``, and
       :math:`u \cdot v` is the dot product of ``u`` and ``v``.

    7. ``Y = pdist(X, 'correlation')``

       Computes the correlation distance between vectors u and v. This is

       .. math::

          1 - \frac{(u - \bar{u}) \cdot (v - \bar{v})}
                   {{||(u - \bar{u})||}_2 {||(v - \bar{v})||}_2}

       where :math:`\bar{v}` is the mean of the elements of vector v,
       and :math:`x \cdot y` is the dot product of :math:`x` and :math:`y`.

    8. ``Y = pdist(X, 'hamming')``

       Computes the normalized Hamming distance, or the proportion of
       those vector elements between two n-vectors ``u`` and ``v``
       which disagree. To save memory, the matrix ``X`` can be of type
       boolean.

    9. ``Y = pdist(X, 'jaccard')``

       Computes the Jaccard distance between the points. Given two
       vectors, ``u`` and ``v``, the Jaccard distance is the
       proportion of those elements ``u[i]`` and ``v[i]`` that
       disagree where at least one of them is non-zero.

    10. ``Y = pdist(X, 'chebyshev')``

       Computes the Chebyshev distance between the points. The
       Chebyshev distance between two n-vectors ``u`` and ``v`` is the
       maximum norm-1 distance between their respective elements. More
       precisely, the distance is given by

       .. math::

          d(u,v) = \max_i {|u_i-v_i|}.

    11. ``Y = pdist(X, 'canberra')``

       Computes the Canberra distance between the points. The
       Canberra distance between two points ``u`` and ``v`` is

       .. math::

         d(u,v) = \sum_i \frac{|u_i-v_i|}
                              {|u_i|+|v_i|}.


    12. ``Y = pdist(X, 'braycurtis')``

       Computes the Bray-Curtis distance between the points. The
       Bray-Curtis distance between two points ``u`` and ``v`` is


       .. math::

            d(u,v) = \frac{\sum_i {u_i-v_i}}
                          {\sum_i {u_i+v_i}}

    13. ``Y = pdist(X, 'mahalanobis', VI=None)``

       Computes the Mahalanobis distance between the points. The
       Mahalanobis distance between two points ``u`` and ``v`` is
       :math:`(u-v)(1/V)(u-v)^T` where :math:`(1/V)` (the ``VI``
       variable) is the inverse covariance. If ``VI`` is not None,
       ``VI`` will be used as the inverse covariance matrix.

    14. ``Y = pdist(X, 'yule')``

       Computes the Yule distance between each pair of boolean
       vectors. (see yule function documentation)

    15. ``Y = pdist(X, 'matching')``

       Computes the matching distance between each pair of boolean
       vectors. (see matching function documentation)

    16. ``Y = pdist(X, 'dice')``

       Computes the Dice distance between each pair of boolean
       vectors. (see dice function documentation)

    17. ``Y = pdist(X, 'kulsinski')``

       Computes the Kulsinski distance between each pair of
       boolean vectors. (see kulsinski function documentation)

    18. ``Y = pdist(X, 'rogerstanimoto')``

       Computes the Rogers-Tanimoto distance between each pair of
       boolean vectors. (see rogerstanimoto function documentation)

    19. ``Y = pdist(X, 'russellrao')``

       Computes the Russell-Rao distance between each pair of
       boolean vectors. (see russellrao function documentation)

    20. ``Y = pdist(X, 'sokalmichener')``

       Computes the Sokal-Michener distance between each pair of
       boolean vectors. (see sokalmichener function documentation)

    21. ``Y = pdist(X, 'sokalsneath')``

       Computes the Sokal-Sneath distance between each pair of
       boolean vectors. (see sokalsneath function documentation)

    22. ``Y = pdist(X, f)``

       Computes the distance between all pairs of vectors in X
       using the user supplied 2-arity function f. For example,
       Euclidean distance between the vectors could be computed
       as follows::

         dm = pdist(X, lambda u, v: np.sqrt(((u-v)**2).sum()))

       Note that you should avoid passing a reference to one of
       the distance functions defined in this library. For example,::

         dm = pdist(X, sokalsneath)

       would calculate the pair-wise distances between the vectors in
       X using the Python function sokalsneath. This would result in
       sokalsneath being called :math:`{n \choose 2}` times, which
       is inefficient. Instead, the optimized C version is more
       efficient, and we call it using the following syntax.::

         dm = pdist(X, 'sokalsneath')

    Parameters
    ----------
    X : ndarray
        An m by n array of m original observations in an
        n-dimensional space.
    metric : string or function
        The distance metric to use. The distance function can
        be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
        'jaccard', 'kulsinski', 'mahalanobis', 'matching',
        'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.
    w : ndarray
        The weight vector (for weighted Minkowski).
    p : double
        The p-norm to apply (for Minkowski, weighted and unweighted)
    V : ndarray
            The variance vector (for standardized Euclidean).
    VI : ndarray
        The inverse of the covariance matrix (for Mahalanobis).

    Returns
    -------
    Y : ndarray
        A condensed distance matrix.

    See Also
    --------
    scipy.spatial.distance.squareform : converts between condensed distance
        matrices and square distance matrices.
    """
    
    X = np.asarray(X, order='c')
    
    def ensure_contiguous(mtx):
        if not mtx.flags.contiguous:
            raise Exception('Prepared trajectories need to be contiguous.')
    def ensure_double(mtx):
        if not mtx.dtype == np.double:
            raise TypeError('Must be of type np.double')
    def ensure_bool(mtx):
        if not mtx.dtype == np.bool:
            raise TypeError('Must be of type np.bool')

    ensure_contiguous(X)
    
    if not ((X.dtype == np.double) or (X.dtype == np.bool)):
        raise TypeError('The vector need to be of type np.double or np.bool.')
    
    s = X.shape
    if len(s) != 2:
        raise ValueError('A 2-dimensional array must be passed.')

    m, n = s
    dm = np.zeros((m * (m - 1) / 2,), dtype=np.double)


    if isinstance(metric, basestring):
        mstr = metric.lower()

        #if X.dtype != np.double and \
        #       (mstr != 'hamming' and mstr != 'jaccard'):
        #    TypeError('A double array must be passed.')
        if mstr in set(['euclidean', 'euclid', 'eu', 'e']):
            ensure_double(X)
            _distance_wrap.pdist_euclidean_wrap(X, dm)
        elif mstr in set(['sqeuclidean', 'sqe', 'sqeuclid']):
            ensure_double(X)
            _distance_wrap.pdist_euclidean_wrap(X, dm)
            dm = dm ** 2.0
        elif mstr in set(['cityblock', 'cblock', 'cb', 'c']):
            ensure_double(X)
            _distance_wrap.pdist_city_block_wrap(X, dm)
        elif mstr in set(['hamming', 'hamm', 'ha', 'h']):
            if X.dtype == np.bool:
                _distance_wrap.pdist_hamming_bool_wrap(X, dm)
            else:
                ensure_double(X)
                _distance_wrap.pdist_hamming_wrap(X, dm)
        elif mstr in set(['jaccard', 'jacc', 'ja', 'j']):
            if X.dtype == np.bool:
                _distance_wrap.pdist_jaccard_bool_wrap(X, dm)
            else:
                ensure_double(X)
                _distance_wrap.pdist_jaccard_wrap(X, dm)
        elif mstr in set(['chebychev', 'chebyshev', 'cheby', 'cheb', 'ch']):
            ensure_double(X)
            _distance_wrap.pdist_chebyshev_wrap(X, dm)
        elif mstr in set(['minkowski', 'mi', 'm']):
            ensure_double(X)
            _distance_wrap.pdist_minkowski_wrap(X, dm, p)

        elif mstr in set(['seuclidean', 'se', 's']):
            ensure_double(X)
            if V is not None:
                V = np.asarray(V, order='c')
                ensure_contiguous(V)
                if type(V) != np.ndarray:
                    raise TypeError('Variance vector V must be a numpy array')
                if V.dtype != np.double:
                    raise TypeError('Variance vector V must contain doubles.')
                if len(V.shape) != 1:
                    raise ValueError('Variance vector V must '
                                     'be one-dimensional.')
                if V.shape[0] != n:
                    raise ValueError('Variance vector V must be of the same '
                            'dimension as the vectors on which the distances '
                            'are computed.')
            else:
                V = np.var(X, axis=0, ddof=1)
            _distance_wrap.pdist_seuclidean_wrap(X, V, dm)
        elif mstr in set(['cosine', 'cos']):
            ensure_double(X)
            norms = np.sqrt(np.sum(X * X, axis=1))
            _distance_wrap.pdist_cosine_wrap(X, dm, norms)
        elif mstr in set(['correlation', 'co']):
            X2 = X - X.mean(1)[:, np.newaxis]
            #X2 = X - np.matlib.repmat(np.mean(X, axis=1).reshape(m, 1), 1, n)
            norms = np.sqrt(np.sum(X2 * X2, axis=1))
            _distance_wrap.pdist_cosine_wrap(X2, dm, norms)
        elif mstr in set(['mahalanobis', 'mahal', 'mah']):
            if VI is not None:
                VI = np.asarray(VI, order='c')
                ensure_contiguous(VI)
                if type(VI) != np.ndarray:
                    raise TypeError('VI must be a numpy array.')
                if VI.dtype != np.double:
                    raise TypeError('The array must contain 64-bit floats.')
            else:
                V = np.cov(X.T)
                VI = np.linalg.inv(V).T.copy()
            # (u-v)V^(-1)(u-v)^T
            ensure_double(X)
            _distance_wrap.pdist_mahalanobis_wrap(X, VI, dm)
        elif mstr == 'canberra':
            ensure_double(X)
            _distance_wrap.pdist_canberra_wrap(X, dm)
            raise ValueError('This is known buggy!')
        elif mstr == 'braycurtis':
            ensure_double(X)
            _distance_wrap.pdist_bray_curtis_wrap(X, dm)
        elif mstr == 'yule':
            ensure_bool(X)
            _distance_wrap.pdist_yule_bool_wrap(X, dm)
        elif mstr == 'matching':
            ensure_bool(X)
            _distance_wrap.pdist_matching_bool_wrap(X, dm)
        elif mstr == 'kulsinski':
            ensure_bool(X)
            _distance_wrap.pdist_kulsinski_bool_wrap(X, dm)
        elif mstr == 'dice':
            ensure_bool(X)
            _distance_wrap.pdist_dice_bool_wrap(X, dm)
        elif mstr == 'rogerstanimoto':
            ensure_bool(X)
            _distance_wrap.pdist_rogerstanimoto_bool_wrap(X, dm)
        elif mstr == 'russellrao':
            ensure_bool(X)
            _distance_wrap.pdist_russellrao_bool_wrap(X, dm)
        elif mstr == 'sokalmichener':
            ensure_bool(X)
            _distance_wrap.pdist_sokalmichener_bool_wrap(X, dm)
        elif mstr == 'sokalsneath':
            ensure_bool(X)
            _distance_wrap.pdist_sokalsneath_bool_wrap(X, dm)
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier')

    return dm


if USE_FAST_CDIST:
    cdist = fast_cdist
    pdist = fast_pdist
else:
    cdist = scipy.spatial.distance.cdist
    pdist = scipy.spatial.distance.pdist
#print 'in metrics library, USE_FAST_CDIST is set to', USE_FAST_CDIST

class AbstractDistanceMetric(object):
    """Abstract base class for distance metrics. All distance metrics should
    inherit from this abstract class.
    
    Provides a niave implementation of all_pairwise and one_to_many in terms
    of the abstract method one_to_all, which may be overridden by subclasses.
    """
    
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def prepare_trajectory(self, trajectory):
        """Prepare trajectory on a format that is more conventient to take
        distances on.
        
        Parameters
        ----------
        trajecory : msmbuilder.Trajectory
            Trajectory to prepare

        Returns
        -------
        prepared_traj : array-like
            the exact form of the prepared_traj is subclass specific, but it should
            support fancy indexing
        
        Notes
        -----
        For RMSD, this is going to mean making word-aligned padded
        arrays (TheoData) suitable for faste calculation, for dihedral-space
        distances means computing the dihedral angles, etc."""
        
        return
        
    
    @abc.abstractmethod
    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        """Calculate the vector of distances from the index1th frame of
        prepared_traj1 to all of the frames in prepared_traj2.
        
        Parameters
        ----------
        prepared_traj1 : prepared_trajectory
            First prepared trajectory
        prepared_traj2 : prepared_trajectory
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory` 
            
        Returns
        -------
        distances : ndarray
            vector of distances of length len(prepared_traj2)
        
        Notes
        -----
        Although this might seem to be a special case of one_to_many(), it
        can often be implemented in a much more optimized way because it doesn't
        require construction of the indices2 array and array slicing in python
        is kindof slow.
        """
        
        return
        
    
    def one_to_many(self, prepared_traj1, prepared_traj2, index1, indices2):
        """Calculate the a vector of distances from the index1th frame of
        prepared_traj1 to all of the indices2 frames of prepared_traj2.
        
        Parameters
        ----------
        prepared_traj1 : prepared_trajectory
            First prepared trajectory
        prepared_traj2 : prepared_trajectory
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory`
        indices2 : ndarray
            list of indices in `prepared_traj2` to calculate the distances to
        
        Returns
        -------
            Vector of distances of length len(indices2)
        
        Notes
        -----
        A subclass should be able to provide a more efficient implementation of
        this
        """
        
        return self.one_to_all(prepared_traj1, prepared_traj2[indices2], index1)
        
    
    def all_pairwise(self, prepared_traj):
        """Calculate condensed distance metric of all pairwise distances
        
        See `scipy.spatial.distance.squareform` for information on how to convert
        the condensed distance matrix to a redundant square matrix
        
        Parameters
        ----------
        prepared_traj : array_like
            Prepared trajectory
        
        Returns
        -------
        Y : ndarray
            A 1D array containing the distance from each frame to each other frame
            
        See Also
        --------
        fast_pdist
        scipy.spatial.distance.squareform
        """
        
        traj_length = len(prepared_traj)
        output = -1 * np.ones(traj_length * (traj_length - 1) / 2)
        p = 0
        for i in xrange(traj_length):
            cmp_indices = np.arange(i + 1, traj_length)
            output[p: p + len(cmp_indices)] = self.one_to_many(prepared_traj, prepared_traj, i, cmp_indices)
            p += len(cmp_indices)
        return output


class RMSD(AbstractDistanceMetric):
    """
    Compute distance between frames using the Room Mean Square Deviation
    over a specifiable set of atoms using the Theobald QCP algorithm
    
    References
    ----------
    .. [1] Theobald, D. L. Acta. Crystallogr., Sect. A 2005, 61, 478-480.
    
    """
    
    class TheoData(object):
        """Stores temporary data required during Theobald RMSD calculation.
        
        Notes:
        Storing temporary data allows us to avoid re-calculating the G-Values
        repeatedly. Also avoids re-centering the coordinates."""
        
        Theoslice = namedtuple('TheoSlice', ('xyz', 'G'))
        def __init__(self, XYZData, NumAtoms=None, G=None):
            """Create a container for intermediate values during RMSD Calculation.
            
            Notes:
            1.  We remove center of mass.
            2.  We pre-calculate matrix magnitudes (ConfG)"""
            
            if NumAtoms is None or G is None:
                NumConfs = len(XYZData)
                NumAtoms = XYZData.shape[1]
                
                self.centerConformations(XYZData)
            
                NumAtomsWithPadding = 4 + NumAtoms - NumAtoms % 4
            
                # Load data and generators into aligned arrays
                XYZData2 = np.zeros((NumConfs, 3, NumAtomsWithPadding), dtype=np.float32)
                for i in range(NumConfs):
                    XYZData2[i, 0:3, 0:NumAtoms] = XYZData[i].transpose()
                
                #Precalculate matrix magnitudes
                ConfG = np.zeros((NumConfs,),dtype=np.float32)
                for i in xrange(NumConfs):
                    ConfG[i] = self.calcGvalue(XYZData[i, :, :])
                
                self.XYZData = XYZData2
                self.G = ConfG
                self.NumAtoms = NumAtoms
                self.NumAtomsWithPadding = NumAtomsWithPadding
                self.CheckCentered()
            else:
                self.XYZData = XYZData
                self.G = G
                self.NumAtoms = NumAtoms
                self.NumAtomsWithPadding = XYZData.shape[2]
            
        
        def __getitem__(self, key):
            # to keep the dimensions right, we make everything a slice
            if isinstance(key, int):
                key = slice(key, key+1)
            return RMSD.TheoData(self.XYZData[key], NumAtoms=self.NumAtoms, G=self.G[key])
        
        def __setitem__(self, key, value):
            self.XYZData[key] = value.XYZData
            self.G[key] = value.G
            
        
        def CheckCentered(self, Epsilon=1E-5):
            """Raise an exception if XYZAtomMajor has nonnzero center of mass(CM)."""
            
            XYZ = self.XYZData.transpose(0, 2, 1)
            x = np.array([max(abs(XYZ[i].mean(0))) for i in xrange(len(XYZ))]).max()
            if x > Epsilon:
                raise Exception("The coordinate data does not appear to have been centered correctly.")
        
        @staticmethod
        def centerConformations(XYZList):
            """Remove the center of mass from conformations.  Inplace to minimize mem. use."""
            
            for ci in xrange(XYZList.shape[0]):
                X = XYZList[ci].astype('float64')#To improve the accuracy of RMSD, it can help to do certain calculations in double precision.
                X -= X.mean(0)
                XYZList[ci] = X.astype('float32')
            return
        
        @staticmethod
        def calcGvalue(XYZ):
            """Calculate the sum of squares of the key matrix G.  A necessary component of Theobold RMSD algorithm."""
            
            conf=XYZ.astype('float64')#Doing this operation in double significantly improves numerical precision of RMSD
            G = 0
            G += np.dot(conf[:, 0], conf[:, 0])
            G += np.dot(conf[:, 1], conf[:, 1])
            G += np.dot(conf[:, 2], conf[:, 2])
            return G
            
        def __len__(self):
            return len(self.XYZData)
    
    
    def __init__(self, atomindices=None, omp_parallel=True):
        """Initalize an RMSD calculator
        
        Parameters
        ----------
        atomindices : array_like, optional
            List of the indices of the atoms that you want to use for the RMSD
            calculation. For example, if your trajectory contains the coordinates
            of all the atoms, but you only want to compute the RMSD on the C-alpha
            atoms, then you can supply a reduced set of atom_indices. If unsupplied,
            all of the atoms will be used.
        omp_parallel : bool, optional
            Use OpenMP parallelized C code under the hood to take advantage of
            multicore architectures. If you're using another parallelization scheme
            (e.g. MPI), you might consider turning off this flag.
            
        Notes
        -----
        You can also control the degree of parallelism with the OMP_NUM_THREADS
        envirnoment variable
            
        
        """
        self.atomindices = atomindices
        self.omp_parallel = omp_parallel
    
    def __repr__(self):
        try:
            val = 'metrics.RMSD(atom_indices=%s, omp_parallel=%s)' % (repr(list(self.atomindices)), self.omp_parallel)
        except:
            val = 'metrics.RMSD(atom_indices=%s, omp_parallel=%s)' % (self.atomindices, self.omp_parallel)
        return val
        
    def prepare_trajectory(self, trajectory):
        """Prepare the trajectory for RMSD calculation.
        
        Preprocessing includes extracting the relevant atoms, centering the
        frames, and computing the G matrix.
        
        
        Parameters
        ----------
        trajectory : msmbuilder.Trajectory
            Molecular dynamics trajectory
        
        Returns
        -------
        theodata : array_like
            A msmbuilder.metrics.TheoData object, which contains some preprocessed
            calculations for the RMSD calculation
        """
        
        if self.atomindices is not None:
            return self.TheoData(trajectory['XYZList'][:, self.atomindices])
        return self.TheoData(trajectory['XYZList'])
    
    
    def one_to_many(self, prepared_traj1, prepared_traj2, index1, indices2):
        """Calculate a vector of distances from one frame of the first trajectory
        to many frames of the second trajectory
        
        The distances calculated are from the `index1`th frame of `prepared_traj1`
        to the frames in `prepared_traj2` with indices `indices2`
        
        Parameters
        ----------
        prepared_traj1 : rmsd.TheoData
            First prepared trajectory
        prepared_traj2 : rmsd.TheoData
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory`
        indices2 : ndarray
            list of indices in `prepared_traj2` to calculate the distances to
        
        Returns
        -------
        Vector of distances of length len(indices2)
        
        Notes
        -----
        If the omp_parallel optional argument is True, we use shared-memory
        parallelization in C to do this faster. Using omp_parallel = False is
        advised if indices2 is a short list and you are paralellizing your
        algorithm (say via mpi) at a different
        level.
        """
        
        if isinstance(indices2, list):
            indices2 = np.array(indices2)
        if not isinstance(prepared_traj1, RMSD.TheoData):
            raise TypeError('Theodata required')
        if not isinstance(prepared_traj2, RMSD.TheoData):
            raise TypeError('Theodata required')
        
        
        if self.omp_parallel:
            return _rmsdcalc.getMultipleRMSDs_aligned_T_g_at_indices(
                      prepared_traj1.NumAtoms, prepared_traj1.NumAtomsWithPadding,
                      prepared_traj1.NumAtomsWithPadding, prepared_traj2.XYZData,
                      prepared_traj1.XYZData[index1], prepared_traj2.G,
                      prepared_traj1.G[index1], indices2)
        else:
            return _rmsdcalc.getMultipleRMSDs_aligned_T_g_at_indices_serial(
                      prepared_traj1.NumAtoms, prepared_traj1.NumAtomsWithPadding,
                      prepared_traj1.NumAtomsWithPadding, prepared_traj2.XYZData,
                      prepared_traj1.XYZData[index1], prepared_traj2.G,
                      prepared_traj1.G[index1], indices2)
        
    
    
    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        """Calculate a vector of distances from one frame of the first trajectory
        to all of the frames in the second trajectory
        
        The distances calculated are from the `index1`th frame of `prepared_traj1`
        to the frames in `prepared_traj2` 
        
        Parameters
        ----------
        prepared_traj1 : rmsd.TheoData
            First prepared trajectory
        prepared_traj2 : rmsd.TheoData
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory`
        
        Returns
        -------
        Vector of distances of length len(prepared_traj2)
        
        Notes
        -----
        If the omp_parallel optional argument is True, we use shared-memory
        parallelization in C to do this faster.
        """
        
        if self.omp_parallel: 
            return _rmsdcalc.getMultipleRMSDs_aligned_T_g(
                prepared_traj1.NumAtoms, prepared_traj1.NumAtomsWithPadding,
                prepared_traj1.NumAtomsWithPadding, prepared_traj2.XYZData,
                prepared_traj1.XYZData[index1], prepared_traj2.G,
                prepared_traj1.G[index1])
        else:            
            return _rmsdcalc.getMultipleRMSDs_aligned_T_g_serial(
                    prepared_traj1.NumAtoms, prepared_traj1.NumAtomsWithPadding,
                    prepared_traj1.NumAtomsWithPadding, prepared_traj2.XYZData,
                    prepared_traj1.XYZData[index1], prepared_traj2.G,
                    prepared_traj1.G[index1])
    
    
    def _square_all_pairwise(self, prepared_traj):
        """Reference implementation of all_pairwise"""
        warnings.warn('This is HORRIBLY inefficient. This operation really needs to be done directly in C')
        traj_length = prepared_traj.XYZData.shape[0]
        output = np.empty((traj_length, traj_length))
        for i in xrange(traj_length):
            output[i] = self.one_to_all(prepared_traj, prepared_traj, i)
        return output
    
    
class Vectorized(AbstractDistanceMetric):
    """Represent MSM frames as vectors in some arbitrary vector space, and then
    use standard vector space metrics. 
    
    Some examples of this might be extracting the contact map or dihedral angles.
    
    In order to be a full featured DistanceMetric, a subclass of
    Vectorized implements its own prepared_trajectory() method, Vectorized
    provides the remainder.
    
    allowable_scipy_metrics gives the list of metrics which your client
    can use. If the vector space that you're projecting your trajectory onto is 
    just a space of boolean vectors, then you probably don't want to allow eulcidean
    distance for instances.
    
    default_scipy_metric is the metric that will be used by your default metric
    if the user leaves the 'metric' field blank/unspecified.
    
    default_scipy_p is the default value of 'p' that will be used if left 
    unspecified. the value 'p' is ONLY used for the minkowski (pnorm) metric, so
    otherwise the scipy.spatial.distance code ignores it anyways.
    
    See http://docs.scipy.org/doc/scipy/reference/spatial.distance.html for a
    description of all the distance metrics and how they work.
    """
    
    allowable_scipy_metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                               'correlation', 'cosine', 'euclidean', 'minkowski',
                               'sqeuclidean','dice', 'kulsinki', 'matching',
                               'rogerstanimoto', 'russellrao', 'sokalmichener',
                               'sokalsneath', 'yule', 'seuclidean', 'mahalanobis',
                               'sqmahalanobis']
    
    def __init__(self, metric='euclidean', p=2, V=None, VI=None):
        """Create a Vectorized metric
        
        Parameters
        ----------
        metric : {'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'euclidean', 'minkowski', 'sqeuclidean','dice', 'kulsinki', 'matching', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'yule', 'seuclidean', 'mahalanobis', 'sqmahalanobis'}
            Distance metric to equip the vector space with.
            See http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
            for details
        p : int, optional
            p-norm order, used for metric='minkowski'
        V : ndarray, optional
            variances, used for metric='seuclidean'
        VI : ndarray, optional
            inverse covariance matrix, used for metric='mahalanobis'
        
        """
        
        self._validate_scipy_metric(metric)
        self.metric = metric
        self.p = p
        self.V = V
        self.VI = VI
        
        if self.metric == 'seuclidean' and V is None:
            raise ValueError('To use seuclidean, you need to supply V')
        if self.metric in ['mahalanobis', 'sqmahalanobis'] and VI is None:
            raise ValueError('To used mahalanobis or sqmahalanobis, you need to supply VI')
        
    
    def _validate_scipy_metric(self, metric):
        """Ensure that "metric" is an "allowable" metric (in allowable_scipy_metrics)"""
        if not metric in self.allowable_scipy_metrics:
            raise TypeError('%s is an  unrecognize metric. "metric" must be one of %s' % (metric, str(self.allowable_scipy_metrics)))
            
    
    def one_to_many(self, prepared_traj1, prepared_traj2, index1, indices2):
        """Calculate a vector of distances from one frame of the first trajectory
        to many frames of the second trajectory
        
        The distances calculated are from the `index1`th frame of `prepared_traj1`
        to the frames in `prepared_traj2` with indices `indices2`
        
        Parameters
        ----------
        prepared_traj1 : ndarray
            First prepared trajectory
        prepared_traj2 : ndarray
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory`
        indices2 : ndarray
            list of indices in `prepared_traj2` to calculate the distances to
        
        Returns
        -------
        distances : ndarray
            Vector of distances of length len(indices2)
        """

        if not isinstance(index1, int):
            raise TypeError('index1 must be of type int.')
        out = cdist(prepared_traj2[indices2], prepared_traj1[[index1]],
                    metric=self.metric, p=self.p, V=self.V, VI=self.VI)
                    
        return out[:, 0]
    
    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        """Measure the distance from one frame to every frame in a trajectory
        
        The distances calculated are from the `index1`th frame of `prepared_traj1`
        to all the frames in `prepared_traj2` with indices `indices2`. Although
        this is similar to one_to_many, it can often be computed faster
        
        Parameters
        ----------
        prepared_traj1 : ndarray
            First prepared trajectory
        prepared_traj2 : ndarray
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory`
        
        
        Returns
        -------
        distances : ndarray
            A vector of distances of length len(prepared_traj2)"""
        
        if not isinstance(index1, int):
            raise TypeError('index1 must be of type int.')
        out2 = cdist(prepared_traj2, prepared_traj1[[index1]], metric=self.metric,
                     p=self.p, V=self.V, VI=self.VI)
        return out2[:, 0]
        
    
    def many_to_many(self, prepared_traj1, prepared_traj2, indices1, indices2):
        """Get a matrix of distances from each frame in a set to each other frame
        in a second set.
        
        Calculate a MATRIX of distances from the frames in prepared_traj1 with
        indices `indices1` to the frames in prepared_traj2 with indices `indices2`,
        using supplied metric.
        
        Parameters
        ----------
        prepared_traj1 : ndarray
            First prepared trajectory
        prepared_traj2 : ndarray
            Second prepared trajectory
        indices1 : array_like
            list of indices in `prepared_traj1` to calculate the distances from
        indices2 : array_like
            list of indices in `prepared_traj2` to calculate the distances to
        
        Returns
        -------
        distances : ndarray
            A 2D array of shape len(indices1) * len(indices2)"""
        
        
        out = cdist(prepared_traj1[indices1], prepared_traj2[indices2], metric=self.metric,
                    p=self.p, V=self.V, VI=self.VI)
        return out
        
    def all_to_all(self, prepared_traj1, prepared_traj2):
        """Get a matrix of distances from all frames in one traj to all frames in
        another
        
        
        Parameters
        ----------
        prepared_traj1 : ndarray
            First prepared trajectory
        prepared_traj2 : ndarray
            Second prepared trajectory
        
        Returns
        -------
        distances : ndarray
            A 2D array of shape len(preprared_traj1) * len(preprared_traj2)"""
        
        if prepared_traj1 is prepared_traj2:
            warnings.warn('runtime', re.sub("\s+", " ", """it's not recommended to
            use this method to calculate the full pairwise distance matrix for
            one trajectory to itself (as you're doing). Use all_pairwise, which
            will be more efficient if you reall need the results as a 2D matrix
            (why?) then you can always use scipy.spatial.distance.squareform()
            on the output of all_pairwise()""".replace('\n', ' ')))
            
        out = cdist(prepared_traj1, prepared_traj2, metric=self.metric, p=self.p,
                    V=self.V, VI=self.VI)
        return out                                        
        
    def all_pairwise(self, prepared_traj):
        """Calculate a condense" distance matrix of all the pairwise distances
        between each frame with each other frame in prepared_traj
        
        The condensed distance matrix can be converted to the redundant square form
        if desired
        
        Parameters
        ----------
        prepared_traj1 : ndarray
            Prepared trajectory
            
        Returns
        -------
        distances : ndarray
            1D vector of length len(pairwise_traj) choose 2 where the i*jth
            entry contains the distance between prepared_traj[i] and prepared_traj[j]
            
        See Also
        --------
        scipy.spatial.distance.pdist
        scipy.spatial.distance.squareform
        """
        
        out = pdist(prepared_traj, metric=self.metric, p=self.p,
                    V=self.V, VI=self.VI)
        return out


class Dihedral(Vectorized, AbstractDistanceMetric):
    """Distance metric for calculating distances between frames based on their
    projection in dihedral space."""
    
    allowable_scipy_metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                               'correlation', 'cosine', 'euclidean', 'minkowski',
                               'sqeuclidean', 'seuclidean', 'mahalanobis', 'sqmahalanobis']
    
    def __init__(self, metric='euclidean', p=2, angles='phi/psi', V=None, VI=None,
        indices=None):
        """Create a distance metric to act on torison angles
        
        Parameters
        ----------
        metric : {'braycurtis', 'canberra', 'chebyshev', 'cityblock',
                  'correlation', 'cosine', 'euclidean', 'minkowski',
                  'sqeuclidean', 'seuclidean', 'mahalanobis', 'sqmahalanobis'}
            Distance metric to equip the vector space with.
        angles : {'phi', 'psi', 'chi', 'omega', 'psi/psi', etc...}
            A slash separated list of strings specifying the types of angles to 
            compute per residue. The choices are 'phi', 'psi', 'chi', and 'omega',
            or any combination thereof
        p : int, optional
            p-norm order, used for metric='minkowski'
        V : ndarray, optional
            variances, used for metric='seuclidean'
        VI : ndarray, optional
            inverse covariance matrix, used for metric='mahalanobi'
        indices : ndarray, optional
            N x 4 numpy array of indices to be considered as dihedral angles. If
            provided, this overrrides the angles argument. The semantics of the
            array are that each row, indices[i], is an array of length 4 giving
            (in order) the indices of 4 atoms that together form a dihedral you
            want to monitor.
        
        See Also
        --------
        fast_cdist
        fast_pdist
        scipy.spatial.distance
        
        """
        super(Dihedral, self).__init__(metric, p, V, VI)
        self.angles = angles
        self.indices = indices
        
        if indices is not None:
            if not isinstance(indices, np.ndarray):
                raise ValueError('indices must be a numpy array')
            if not indices.ndim == 2:
                raise ValueError('indices must be 2D')
            if not indices.dtype == np.int:
                raise ValueError('indices must contain ints')
            if not indices.shape[1] == 4:
                raise ValueError('indices must be N x 4')
            logger.warning('OVERRIDING angles=%s and using custom indices instead', angles)
    
    def __repr__(self):
        "String representation of the object"
        return 'metrics.Dihedral(metric=%s, p=%s, angles=%s)' % (self.metric, self.p, self.angles)
    
    def prepare_trajectory(self, trajectory):
        """Prepare the dihedral angle representation of a trajectory, suitable
        for distance calculations.
        
        Parameters
        ----------
        trajectory : msmbuilder.Trajectory
            An MSMBuilder trajectory to prepare
        
        Returns
        -------
        projected_angles : ndarray
            A 2D array of dimension len(trajectory) x (2*number of dihedral
            angles per frame), such that in each row, the first half of the entries
            contain the cosine of the dihedral angles and the later dihedral angles
            contain the sine of the dihedral angles. This transform is necessary so
            that distance calculations preserve the periodic symmetry.
        """
        
        traj_length = len(trajectory['XYZList'])
        
        if self.indices is None:
            indices = _dihedralcalc.get_indices(trajectory, self.angles)
        else:
            indices = self.indices
        
        dihedrals = _dihedralcalc.compute_dihedrals(trajectory, indices, degrees=False)
        
        # these dihedrals go between -pi and pi but obviously because of the
        # periodicity, when we take distances we want the distance between -179
        # and +179 to be very close, so we need to do a little transform
        
        num_dihedrals = dihedrals.shape[1]
        transformed = np.empty((traj_length, 2 * num_dihedrals))
        transformed[:, 0:num_dihedrals] = np.cos(dihedrals)
        transformed[:, num_dihedrals:2*num_dihedrals] = np.sin(dihedrals)
        
        return np.double(transformed)
        
    


class ContinuousContact(Vectorized, AbstractDistanceMetric):
    """Distance metric for calculating distances between frames based on the
    pairwise distances between residues.
    
    Here each frame is represented as a vector of the distances between pairs
    of residues.
    """
    
    allowable_scipy_metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                               'correlation', 'cosine', 'euclidean', 'minkowski',
                               'sqeuclidean']
    
    def __init__(self, metric='euclidean', p=2, contacts='all', scheme='closest-heavy'):
        """Create a distance calculator based on the distances between pairs of atoms
        in a sturcture -- like the contact map except without casting to boolean.
        
        Parameters
        ----------
        metric : {'braycurtis', 'canberra', 'chebyshev', 'cityblock',
                  'correlation', 'cosine', 'euclidean', 'minkowski',
                  'sqeuclidean'}
            distance metric to equip the space with
        p : int
            exponent for p-norm, used only for `metric='minkowski'`
        contacts : {ndarray, 'all'}
            contacts can be an n by 2 array, where each row is a pair of
            integers giving the indices of 2 residues whose distance you care about.
            Alternatively, contacts can be the string 'all'. This is a shortcut for
            supplying a contacts list that includes all (N-2 * N-3) / 2 pairs of
            residues which are more than 2 residues apart.
        scheme: {'CA', 'closest', 'closest-heavy'}
            scheme can be 'CA', 'closest', or 'closest-heavy' and gives
            the sense in which the 'distance between two residues' is computed. If
            scheme is 'CA', then we'll use the cartesian distance between the residues'
            C-alpha atoms as their distance for the purpose of calculating whether or
            not they have exceeded the cutoff. If scheme is 'closest', we'll use the
            distance between the closest pair of atoms where one
            belongs to residue i and to residue j. If scheme is 'closest-heavy', we'll
            use the distance between the closest pair of non-hydrogen atoms where one
            belongs to reside i and one to residue j.
        """
        
        super(ContinuousContact, self).__init__(metric, p)
        self.contacts = contacts
        
        scheme = scheme.lower()
        if not scheme in ['ca', 'closest', 'closest-heavy']:
            raise ValueError('Unrecognized scheme')
        
        self.scheme = scheme
        
    
    def __repr__(self):
        try:
            contacts_repr = repr(self.contacts.tolist())
        except:
            contacts_repr = repr(self.contacts)
        return 'metrics.ContinuousContact(metric=%s, p=%s, contacts=%s, scheme=%s)' % (self.metric, self.p, contacts_repr, self.scheme)
        
    
    def prepare_trajectory(self, trajectory):
        """Prepare a trajectory for distance calculations based on the contact map.
        
        Each frame in the trajectory will be represented by a vector where
        each entries represents the distance between two residues in the structure.
        Depending on what contacts you pick to use, this can be a 'native biased' 
        picture or not.
        
        Paramters
        ---------
        trajectory : msmbuilder.Trajectory
            The trajectory to prepare
            
        Returns
        -------
        pairwise_distances : ndarray
            1D array of various residue-residue distances
        """
        
        xyzlist = trajectory['XYZList']
        traj_length = len(xyzlist)
        num_residues = trajectory.GetNumberOfResidues()
        num_atoms = trajectory.GetNumberOfAtoms()
        
        if self.contacts == 'all':
            contacts = np.empty(((num_residues - 2) * (num_residues - 3) / 2, 2), dtype=np.int32)
            p = 0
            for (a, b) in itertools.combinations(range(num_residues), 2):
                if max(a, b) > min(a, b) + 2:
                    contacts[p, :] = [a, b]
                    p += 1
            assert p == len(contacts), 'Something went wrong generating "all"'
            
        else:
            num, width = self.contacts.shape
            contacts = self.contacts
            if not width == 2:
                raise ValueError('contacts must be width 2')
            if not (0 < len(np.unique(contacts[:, 0])) < num_residues):
                raise ValueError('contacts should refer to zero-based indexing of the residues')
            if not np.all(np.logical_and(0 <= np.unique(contacts), np.unique(contacts) < num_residues)):
                raise ValueError('contacts should refer to zero-based indexing of the residues')
                
        if self.scheme == 'ca':
            # not all residues have a CA
            #alpha_indices = np.where(trajectory['AtomNames'] == 'CA')[0]
            atom_contacts = np.zeros_like(contacts)
            residue_to_alpha = np.zeros(num_residues) # zero based indexing
            for i in range(num_atoms):
                if trajectory['AtomNames'][i] == 'CA':
                    residue = trajectory['ResidueID'][i] - 1
                    residue_to_alpha[residue] = i
            #print 'contacts (residues)', contacts
            #print 'residue_to_alpja', residue_to_alpha.shape
            #print 'residue_to_alpja', residue_to_alpha
            atom_contacts = residue_to_alpha[contacts]
            #print 'atom_contacts', atom_contacts
            output = _contactcalc.atom_distances(xyzlist, atom_contacts)
        
        elif self.scheme in ['closest', 'closest-heavy']:
            if self.scheme == 'closest':
                residue_membership = [None for i in range(num_residues)]
                for i in range(num_residues):
                    residue_membership[i] = np.where(trajectory['ResidueID'] == i + 1)[0]
            elif self.scheme == 'closest-heavy':
                 residue_membership = [[] for i in range(num_residues)]
                 for i in range(num_atoms):
                     residue = trajectory['ResidueID'][i] - 1
                     if not trajectory['AtomNames'][i].lstrip('0123456789').startswith('H'):
                         residue_membership[residue].append(i)
            
            #print 'Residue Membership'
            #print residue_membership
            #for row in residue_membership:
            #    for col in row:
            #        print "%s-%s" % (trajectory['AtomNames'][col], trajectory['ResidueID'][col]),
            #    print
            output = _contactcalc.residue_distances(xyzlist, residue_membership, contacts)
        else:
            raise ValueError('This is not supposed to happen!')
            
        return np.double(output)
    

class BooleanContact(Vectorized, AbstractDistanceMetric):
    """Distance metric for calculating distances between frames based on their
    contact maps.
    
    Here each frame is represented as a vector of booleans representing whether
    the distance between pairs of residues is less than a cutoff.
    """
    
    allowable_scipy_metrics = ['dice', 'kulsinki', 'matching', 'rogerstanimoto',
                               'russellrao', 'sokalmichener', 'sokalsneath',
                               'yule']
    
    def __init__(self, metric='matching', contacts='all', cutoff=0.5, scheme='closest-heavy'):
        """ Create a distance metric that will measure the distance between frames
        based on differences in their contact maps.
        
        Paramters
        ---------
        metric : {'dice', 'kulsinki', 'matching', 'rogerstanimoto',
                  russellrao', 'sokalmichener', 'sokalsneath', 'yule'}
            You should probably use matching. Then the distance between two
            frames is just the number of elements in their contact map that are
            the same. See the scipy.spatial.distance documentation for details.
        contacts : {ndarray, 'all'}
            contacts can be an n by 2 array, where each row is a pair of
            integers giving the indices of 2 residues which form a native contact.
            Each conformation is then represnted by a vector of booleans representing
            whether or not that contact is present in the conformation. The distance
            metric acts on two conformations and compares their vectors of booleans.
            Alternatively, contacts can be the string 'all'. This is a shortcut for
            supplying a contacts list that includes all (N-2 * N-3) / 2 pairs of
            residues which are more than 2 residues apart.
        cutoff : {float, ndarray}
            cutoff can be either a positive float representing the cutoff distance between two
            residues which constitues them being 'in contact' vs 'not in contact'. It
            is measured in the same distance units that your trajectory's XYZ data is in
            (probably nanometers).
            Alternatively, cutoff can be an array of length equal to the number of rows in the
            contacts array, specifying a different cutoff for each contact. That is, cutoff[i]
            should contain the cutoff for the contact in contact[i].
        scheme : {'CA', 'closest', 'closest-heavy'}
            scheme can be 'CA', 'closest', or 'closest-heavy' and gives
            the sense in which the 'distance between two residues' is computed. If
            scheme is 'CA', then we'll use the cartesian distance between the residues'
            C-alpha atoms as their distance for the purpose of calculating whether or
            not they have exceeded the cutoff. If scheme is 'closest', we'll use the
            distance between the closest pair of atoms where one belongs to residue i
            and to residue j. If scheme is 'closest-heavy', we'll use the distance
            between the closest pair of non-hydrogen atoms where one belongs to reside
            i and one to residue j."""
        
        super(BooleanContact, self).__init__(metric)
        self.contacts = contacts

        if isinstance( cutoff, Number ):
            self.cutoff = cutoff
        else:
            self.cutoff = np.array( cutoff ).flatten()
      
        scheme = scheme.lower()
        if not scheme in ['ca', 'closest', 'closest-heavy']:
            raise ValueError('Unrecognized scheme')
              
        self.scheme = scheme
        
    
    def __repr__(self):
        try:
            contacts_repr = repr(self.contacts.tolist())
        except:
            contacts_repr = repr(self.contacts)
        
        try:
            cutoff_repr = repr(self.cutoff.tolist())
        except:
            cutoff_repr = repr(self.cutoff)
        
        return 'metrics.BooleanContact(metric=%s, p=%s, contacts=%s, cutoff=%s, scheme=%s)' % (self.metric, self.p, contacts_repr, cutoff_repr, self.scheme)
    
    
    def prepare_trajectory(self, trajectory):
        """Prepare a trajectory for distance calculations based on the contact map.
        
        Paramters
        ---------
        trajectory : msmbuilder.Trajectory
            The trajectory to prepare
            
        Returns
        -------
        pairwise_distances : ndarray
            1D array of various residue-residue distances, casted to boolean
        """

        
        ccm = ContinuousContact(contacts=self.contacts, scheme=self.scheme)
        contact_d = ccm.prepare_trajectory(trajectory)
        if not isinstance(self.cutoff, Number):
            if not len(self.cutoff) == contact_d.shape[1]: # contact_d has frames in rows and contacts in columns
                raise ValueError('cutoff must be a number or match the length of contacts')
    
        #contact = np.zeros_like(contact_d).astype(bool)
        #for i in xrange(contact_d.shape[0]):
        #    contact[i, :] = contact_d[i, :] < self.cutoff
        contact = contact_d < self.cutoff
        return contact


class AtomPairs(Vectorized, AbstractDistanceMetric):
    """Concrete distance metric that monitors the distance
    between certain pairs of atoms (as opposed to certain pairs of residues
    as ContinuousContact does)"""
    
    allowable_scipy_metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                               'correlation', 'cosine', 'euclidean', 'minkowski',
                               'sqeuclidean', 'seuclidean', 'mahalanobis']
                               
    def __init__(self, metric='cityblock', p=1, atom_pairs=None, V=None, VI=None):
        """ Atom pairs should be a N x 2 array of the N pairs of atoms
        whose distance you want to monitor"""
        super(AtomPairs, self).__init__(metric, p, V=V, VI=VI)
        try:
            atom_pairs = np.array(atom_pairs, dtype=int)
            n, m = atom_pairs.shape
            if not m == 2:
                raise ValueError()
        except ValueError:
            raise ValueError('Atom pairs must be an n x 2 array of pairs of atoms')
        self.atom_pairs = np.int32(atom_pairs)
        
    def prepare_trajectory(self, trajectory):
        length = len(trajectory['XYZList'])
        ptraj = _contactcalc.atom_distances(trajectory['XYZList'], self.atom_pairs)
        return np.double(ptraj)
        

class Rg(Vectorized, AbstractDistanceMetric):
    """Concrete distance metric that calculates the distance between frames
    as just their difference in Rg"""
    
    allowable_scipy_metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                               'correlation', 'cosine', 'euclidean', 'minkowski',
                               'sqeuclidean']
                               
    def __init__(self, metric='euclidean', p=2):
        super(Rg, self).__init__(metric, p)
    
    def prepare_trajectory(self, trajectory):
        """Calculate the Rg of every frame"""
        return _rgcalc.calculate_rg(trajectory['XYZList'])
    

class ProtLigRMSD(AbstractDistanceMetric):
    """
    Distance metric for calculating the the state of a protien-ligand system
    
    Each frame is rotated so as to get the protein_indices atoms into
    maximum coincidence with the pdb structure, and then the distance is
    taken as an RMSD between only the ligand_indices.
    
    Note that this is a pure python implementation, so it's not going to be
    fast.

    The same functionality can be done using the LPRMSD module.
    """
    
    def __init__(self, protein_indices, ligand_indices, pdb):
        """protein_indices should be a length n array of the indices
        of the atoms which the rotation will occur with repsect to.
        ligand_indices should be an array of length m of the atoms that
        the RMSD calculation will occur with repsect to.
        
        pdb should be an msmbuilder.Conformation object which contains the XYZ
        coordinates of the protein in its crystal pose. The number of atoms
        in the conformation should be directly equal to the number of atoms in
        protein_indices, and they should be corresponding.
        """
        warnings.warn('Depricated. Consider usings LPRMSD')
        self.protein_indices = protein_indices
        self.ligand_indices = ligand_indices
        if 'XYZ' in pdb:
            self.pdb_xyz = pdb['XYZ']
        elif 'XYZList' in pdb:
            warnings.warn("You passed a Trajectory object for the PDB? I'm taking the first frame")
            self.pdb_xyz = pdb['XYZList'][0, :, :]
        else:
            raise ValueError("Couldn't handle pdb:%s" % s)
        if not len(self.protein_indices) == self.pdb_xyz.shape[0]:
            raise ValueError("There should be the same number of protein indices as there are atoms in the pdb structure")
        
        self.USE_FLOAT_64_FOR_MEAN = True
    
    def prepare_trajectory(self, trajectory):
        """
        Rotate and center each frame in trajectory with repsect to the pdb, then extract
        the ligand indices
        """
        
        xyzlist = trajectory['XYZList']
        traj_length, num_atoms, num_dims = xyzlist.shape
        aligned_ligand_xyz = np.zeros((traj_length, len(self.ligand_indices), 3), dtype=np.float32)

        #for i in xrange(traj_length):
        #    all_xyz = self._centered(xyzlist[i])
        #    rotation_matrix = self._rotatation(all_xyz[self.protein_indices, :], self.pdb_xyz)
        #    ligand_xyz = all_xyz[self.ligand_indices, :]
        #    aligned_ligand_xyz[i] = np.transpose(np.dot(rotation_matrix, np.transpose(ligand_xyz)))
        for i in xrange(traj_length):
            #get all coords for frame i
            all_xyz = (xyzlist[i])
            prot_xyz = all_xyz[self.protein_indices, :]
            prot_pdb = self.pdb_xyz[self.protein_indices]
            ligand_xyz = all_xyz[self.ligand_indices, :]

            # center these by same protein indices         
            if self.USE_FLOAT_64_FOR_MEAN:
                float64_prot_xyz = prot_xyz.astype('float64')
                prot_mean = float64_prot_xyz.mean(0)
                float64_pdb_xyz = prot_pdb.astype('float64')
                pdb_mean = float64_pdb_xyz.mean(0)
            else:
                prot_mean = prot_xyz.mean(0)
                pdb_mean = prot_pdb.mean(0)

            cent_prot_xyz = prot_xyz - prot_mean
            cent_prot_pdb = prot_pdb - pdb_mean
            cent_lig_xyz = ligand_xyz - prot_mean

            #after centering, compute rotation matrix
            rotation_matrix = self._rotation(cent_prot_xyz, cent_prot_pdb) #2nd arg should be pdb
            aligned_ligand_xyz[i] = np.transpose(np.dot(rotation_matrix, np.transpose(cent_lig_xyz)))
            
        return aligned_ligand_xyz
    
    
    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        """Calculate the vector of distances from the index1th frame of
        prepared_traj1 to all of the frames in prepared_traj2.
        
        Parameters
        ----------
        prepared_traj1 : prepared_trajectory
            First prepared trajectory
        prepared_traj2 : prepared_trajectory
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory` 
            
        Returns
        -------
        distances : ndarray
            vector of distances of length len(prepared_traj2)
        
        Notes
        -----
        Although this might seem to be a special case of one_to_many(), it
        can often be implemented in a much more optimized way because it doesn't
        require construction of the indices2 array and array slicing in python
        is kindof slow.
        """
        
        length = len(prepared_traj2)
        distances = np.zeros(length)
        for i in xrange(length):
            distances[i] = self._calcRMSD(prepared_traj1[index1], prepared_traj2[i])
        return distances
    
    @staticmethod
    def _calcRMSD(crds1, crds2):
        """Returns RMSD between 2 sets of [nx3] np array--NOTE this requires pre-removed Center of mass."""
        """ When this is used, I pass in AlignedData, with removed COM"""
        assert(crds1.shape[1] == 3)
        assert(crds1.shape == crds2.shape)
        n_vec = np.shape(crds1)[0]
        correlation_matrix = np.dot(np.transpose(crds1), crds2)
        v, s, w_tr = np.linalg.svd(correlation_matrix)
        is_reflection = (np.linalg.det(v) * np.linalg.det(w_tr)) < 0.0
        if is_reflection:
            s[-1] = - s[-1]
        E0 = sum(sum(crds1 * crds1)) + \
        sum(sum(crds2 * crds2))
        rmsd_sq = (E0 - 2.0 * sum(s)) / float(n_vec)
        rmsd_sq = max([rmsd_sq, 0.0])
        return np.sqrt(rmsd_sq)
    
    @staticmethod
    def _centered(xyz):
        """Return a copied version of xyz with the COM removed
        Note that this does an upconversion to double precision to get the mean,
        although the result is returned in float32"""
        xyz = xyz.copy()
        xyz -= np.float64(xyz).mean(0)
        return np.float32(xyz)
                
    @staticmethod
    def _rotatation(from_xyz, to_xyz):
        """Return a rotated version of *from_xyz* to bring it in maximum cooincidence with
        *to_xyz*
        
        from_xyz and to_xyz should both be n x 3 arrays
        """
        n0, n1 = from_xyz.shape
        n2, n3 = to_xyz.shape
        if not ((n0 == n2) and (n1 == n3) and n1 == 3):
            raise ValueError("Wrong dims")
        
        from_xyz = ProtLigRMSD._centered(from_xyz)
        to_xyz = ProtLigRMSD._centered(to_xyz)
        
        correlation_matrix = np.dot(np.transpose(to_xyz), from_xyz)
        v, s, w_tr = np.linalg.svd(correlation_matrix)
        is_reflection = (np.linalg.det(v) * np.linalg.det(w_tr)) < 0.0
        if is_reflection:
            v[:, -1] = -v[:, -1]
        rotation_matrix = np.dot(v, w_tr)
            
        return rotation_matrix
    

class Hybrid(AbstractDistanceMetric):
    "A linear combination of other distance metrics"
    
    class HybridPreparedTrajectory(object):
        """Container to to hold the prepared trajectory.
        This container needs to support slice notation in a way that kind
        of passes through the indices to the 2nd dimension. So if you
        have a HybridPreparedTrajectory with 3 bases metrics, and you
        do metric[0:100], it needs to return a HybridPrepareTrajectory
        with the same three base metrics, but with each of the base prepared
        trajectories sliced 0:100. We don't want to slice out base_metrics
        and thus have metric[0] return only one of the three prepared_trajectories
        in its full length."""
        def __init__(self, *args):
            self.num_base = len(args)
            self.length = len(args[0])
            if not np.all((len(arg) == self.length for arg in args)):
                raise ValueError("Must all be equal length")
            
            self.datas = args
            
        
        def __getitem__(self, key):
            if isinstance(key, int):
                key = slice(key, key+1)
            return Hybrid.HybridPreparedTrajectory(*(d[key] for d in self.datas))
        
        def __len__(self):
            return self.length
        
        def __setitem__(self, key, value):
            try:
                if self.num_base != value.num_base:
                    raise ValueError("Must be prepared over the same metrics")
            except:
                raise ValueError("I can only set in something which is also a HybridPreparedTrajectory")
            
            for i in xrange(self.num_base):
                self.datas[i][key] = value.datas[i]
            
        
    
    def __init__(self, base_metrics, weights):
        """Create a hybrid linear combinatiin distance metric
        
        Parameters
        ----------
        base_metrics : list of distance metric objects
        weights : list of floats
            list of scalars of equal length to `base_metrics` -- each base
            metric will be multiplied by that scalar when they get summed.
        """
        
        self.base_metrics = base_metrics
        self.weights = weights
        self.num = len(self.base_metrics)
        
        if not len(self.weights) == self.num:
            raise ValueError()
        

    def prepare_trajectory(self, trajectory):
        """Preprocess trajectory for use with this metric
        
        Parameters
        ----------
        trajectory : msmbuilder.Trajectory
            Trajectory to prepare
        
        Returns
        -------
        prepared_trajectory : array_like
            The prepared trajectory is a special array like object called
            HybridPreparedTrajectory which is designed to pass through the slicing
            correctly so that if you ask for prepared_trajectory[5] you get the
            appropriate 5th frames dihedral angles, RMSD, etc (depending what
            base metrics you used)
        """
        prepared = (m.prepare_trajectory(trajectory) for m in self.base_metrics)
        return self.HybridPreparedTrajectory(*prepared)
    

    def one_to_many(self, prepared_traj1, prepared_traj2, index1, indices2):
        """Calculate a vector of distances from one frame of the first trajectory
        to many frames of the second trajectory
        
        The distances calculated are from the `index1`th frame of `prepared_traj1`
        to the frames in `prepared_traj2` with indices `indices2`
        
        Parameters
        ----------
        prepared_traj1 : ndarray
            First prepared trajectory
        prepared_traj2 : ndarray
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory`
        indices2 : ndarray
            list of indices in `prepared_traj2` to calculate the distances to
        
        Returns
        -------
        Vector of distances of length len(indices2)
        """
        distances = None
        for i in range(self.num):
            d = self.base_metrics[i].one_to_many(prepared_traj1.datas[i], prepared_traj2.datas[i], index1, indices2)
            if distances is None:
                distances = self.weights[i] * d
            else:
                distances += self.weights[i] * d
        return distances
    

    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        """Calculate the vector of distances from the index1th frame of
        prepared_traj1 to all of the frames in prepared_traj2.
        
        Parameters
        ----------
        prepared_traj1 : prepared_trajectory
            First prepared trajectory
        prepared_traj2 : prepared_trajectory
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory` 
            
        Returns
        -------
        distances : ndarray
            vector of distances of length len(prepared_traj2)
        
        Notes
        -----
        Although this might seem to be a special case of one_to_many(), it
        can often be implemented in a much more optimized way because it doesn't
        require construction of the indices2 array and array slicing in python
        is kindof slow.
        """
        
        distances = None
        for i in range(self.num):
            d = self.base_metrics[i].one_to_all(prepared_traj1.datas[i], prepared_traj2.datas[i], index1)
            if distances is None:
                distances = self.weights[i] * d
            else:
                distances += self.weights[i] * d
        return distances
    

    def all_pairwise(self, prepared_traj):
        """Calculate condensed distance metric of all pairwise distances
        
        See `scipy.spatial.distance.squareform` for information on how to convert
        the condensed distance matrix to a redundant square matrix
        
        Parameters
        ----------
        prepared_traj : array_like
            Prepared trajectory
        
        Returns
        -------
        Y : ndarray
            A 1D array containing the distance from each frame to each other frame
            
        See Also
        --------
        fast_pdist
        scipy.spatial.distance.squareform
        """
        
        distances = None
        for i in range(self.num):
            d = self.base_metrics[i].all_pairwise(prepared_traj.datas[i])
            distances = self.weights[i] * d if distances is None else distances + self.weights[i] * d
        return distances
    

class HybridPNorm(Hybrid):
    """A p-norm combination of other distance metrics. With p=2 for instance,
    this gives you the root mean square combination of the base metrics"""
    
    def __init__(self, base_metrics, weights, p=2):
        """Initialize the HybridPNorm distance metric.
        
        Parameters
        ----------
        base_metrics : list of distance metric objects
        weights : list of floats
            list of scalars of equal length to `base_metrics` -- each base
            metric will be multiplied by that scalar.
        p : float
            p should be a scalar, greater than 0, which will be the exponent.
            If p=2, all the base metrics will be squared, then summed, then the
            square root will be taken. If p=3, the base metrics will be cubed,
            summed and cube rooted, etc.
        """
        
        self.p = float(p)
        super(HybridPNorm, self).__init__(base_metrics, weights)
        
    
    def one_to_many(self, prepared_traj1, prepared_traj2, index1, indices2):
        """Calculate a vector of distances from one frame of the first trajectory
        to many frames of the second trajectory
        
        The distances calculated are from the `index1`th frame of `prepared_traj1`
        to the frames in `prepared_traj2` with indices `indices2`
        
        Parameters
        ----------
        prepared_traj1 : ndarray
            First prepared trajectory
        prepared_traj2 : ndarray
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory`
        indices2 : ndarray
            list of indices in `prepared_traj2` to calculate the distances to
        
        Returns
        -------
        Vector of distances of length len(indices2)
        """
        
        distances = None
        for i in range(self.num):
            d = self.base_metrics[i].one_to_many(prepared_traj1.datas[i], prepared_traj2.datas[i], index1, indices2)
            if distances is None:
                distances = (self.weights[i]*d)**self.p
            else:
                distances += (self.weights[i]*d)**self.p
        return distances**(1.0 / self.p)
        
    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        """Calculate the vector of distances from the index1th frame of
        prepared_traj1 to all of the frames in prepared_traj2.
        
        Parameters
        ----------
        prepared_traj1 : prepared_trajectory
            First prepared trajectory
        prepared_traj2 : prepared_trajectory
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory` 
            
        Returns
        -------
        distances : ndarray
            vector of distances of length len(prepared_traj2)
        
        Notes
        -----
        Although this might seem to be a special case of one_to_many(), it
        can often be implemented in a much more optimized way because it doesn't
        require construction of the indices2 array and array slicing in python
        is kindof slow.
        """
        
        distances = None
        for i in range(self.num):
            d = self.base_metrics[i].one_to_all(prepared_traj1.datas[i], prepared_traj2.datas[i], index1)
            if distances is None:
                distances = (self.weights[i]*d)**self.p
            else:
                distances += (self.weights[i]*d)**self.p
        return distances**(1.0 / self.p)
        
    def all_pairwise(self, prepared_traj):
        """Calculate condensed distance metric of all pairwise distances
        
        See `scipy.spatial.distance.squareform` for information on how to convert
        the condensed distance matrix to a redundant square matrix
        
        Parameters
        ----------
        prepared_traj : array_like
            Prepared trajectory
        
        Returns
        -------
        Y : ndarray
            A 1D array containing the distance from each frame to each other frame
            
        See Also
        --------
        fast_pdist
        scipy.spatial.distance.squareform
        """
                
        distances = None
        for i in range(self.num):
            d = self.base_metrics[i].all_pairwise(prepared_traj.datas[i])
            d = (self.weights[i]*d)**self.p
            distances = d if distances is None else distances + (self.weights[i]*d)
            logger.info('got %s', i)
        return distances**(1.0 / self.p)

