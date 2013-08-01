import numpy as np
import scipy.spatial.distance
from msmbuilder import _distance_wrap


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

    if len(XA.shape) != 2:
        raise ValueError('XA must be a 2-dimensional array.')
    if len(XB.shape) != 2:
        raise ValueError('XB must be a 2-dimensional array.')
    if XA.shape[1] != XB.shape[1]:
        raise ValueError('XA and XB must have the same number of columns '
                         '(i.e. feature dimension.)')
    
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
