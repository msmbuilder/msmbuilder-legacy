from __future__ import print_function, division, absolute_import

import numpy as np
import re
import types
import warnings
import functools
import collections
from mdtraj.utils.six import PY3
from mdtraj.utils.six.moves import copyreg, xrange
if PY3:
    from itertools import filterfalse as ifilterfalse
else:
    from itertools import ifilterfalse

warnings.simplefilter('always')


def uneven_zip(*args):
    '''Zip the arguments together like the builtin function, except that
    when one argument runs out (because its shorter), you keep filling it in
    with its last value

    i.e.

    uneven_zip([1,2,3], 'a', [10,11]) = [[1, 'a', 10], [2, 'a', 11], [3, 'a', 11]]
    '''
    num_args = len(args)
    args = list(args)
    for i in xrange(num_args):
        if not hasattr(args[i], '__len__'):
            args[i] = (args[i],)
    lengths = [len(arg) for arg in args]

    def get(i):
        result = [None] * num_args
        for j in range(num_args):
            try:
                result[j] = args[j][i]
            except:
                result[j] = args[j][lengths[j] - 1]
        return result
    zipped = [get(i) for i in range(max(lengths))]
    return zipped


def format_block(block):
    '''Format the given block of text, trimming leading/trailing
    empty lines and any leading whitespace that is common to all lines.
    The purpose is to let us list a code block as a multiline,
    triple-quoted Python string, taking care of indentation concerns.'''
    # separate block into lines
    lines = str(block).split('\n')
    # remove leading/trailing empty lines
    while lines and not lines[0]:
        del lines[0]
    while lines and not lines[-1]:
        del lines[-1]
    # look at first line to see how much indentation to trim
    ws = re.match(r'\s*', lines[0]).group(0)
    if ws:
        lines = [x.replace(ws, '', 1) for x in lines]
    # remove leading/trailing blank lines (after leading ws removal)
    # we do this again in case there were pure-whitespace lines
    while lines and not lines[0]:
        del lines[0]
    while lines and not lines[-1]:
        del lines[-1]

    return '\n'.join(lines) + '\n'


def keynat(string):
    '''A natural sort helper function for sort() and sorted()
    without using regular expression.

    >>> items = ('Z', 'a', '10', '1', '9')
    >>> sorted(items)
    ['1', '10', '9', 'Z', 'a']
    >>> sorted(items, key=keynat)
    ['1', '9', '10', 'Z', 'a']
    '''
    r = []
    for c in string:
        try:
            c = int(c)
            try:
                r[-1] = r[-1] * 10 + c
            except:
                r.append(c)
        except:
            r.append(c)
    return r


def _pickle_method(method):
    func_name = method.__func.__name__
    obj = method.__self__
    cls = method.__self__.__class__
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


def make_methods_pickable():
    "Run this at the top of a script to register pickable methods"
    copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)


def deprecated(replacement=None, removal_version=None):
    """A decorator which can be used to mark functions as deprecated.
    replacement is a callable that will be called with the same args
    as the decorated function.

    Code adapted from http://code.activestate.com/recipes/577819-deprecated-decorator/,
    MIT license

    >>> @deprecated()
    ... def foo(x):
    ...     return x
    ...
    >>> ret = foo(1)
    DeprecationWarning: foo is deprecated
    >>> ret
    1
    >>>
    >>>
    >>> def newfun(x):
    ...     return 0
    ...
    >>> @deprecated(newfun)
    ... def foo(x):
    ...     return x
    ...
    >>> ret = foo(1)
    DeprecationWarning: foo is deprecated; use newfun instead
    >>> ret
    0
    >>>
    """
    def outer(oldfun):
        def inner(*args, **kwargs):
            msg = "%s is deprecated use %s instead. " % (oldfun.__name__, replacement.__name__)

            if removal_version is not None:
                msg += '%s will be removed in version %s' % (oldfun.__name__, removal_version)

            warnings.warn(msg, DeprecationWarning, stacklevel=2)

            return replacement(*args, **kwargs)

        return inner
    return outer


def future_warning(func):
    '''This is a decorator which can be used to mark functions
    as to-be deprecated. It will result in a warning being emitted
    when the function is used.'''

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn_explicit(
            "Call to future function {}.".format(func.__name__),
            category=FutureWarning,
            filename=func.__code__.co_filename,
            lineno=func.__code__.co_firstlineno + 1
        )
        return func(*args, **kwargs)
    return new_func


def highlight(text, color='Red', bold=False):
    """Return a highlighted string using color or bold.

    @param[in] text The string that the printout is based upon.  This function
    will return the highlighted string.

    @param[in] color String or number corresponding to the color.
    1 red\n
    2 green\n
    3 yellow\n
    4 blue\n
    5 magenta\n
    6 cyan\n
    7 white

    @param[in] bold Whether to use bold print
    """

    colordict = {'red': 1,
                 'green': 2,
                 'yellow': 3,
                 'blue': 4,
                 'magenta': 5,
                 'cyan': 6,
                 'white': 7}

    if color.lower() in colordict:
        color = colordict[color.lower()]
    elif color in ['1', '2', '3', '4', '5', '6', '7']:
        color = int(color)
    elif color in range(1, 8):
        pass
    else:
        raise ValueError(
            'Invalid argument given for color (use integer 1-7 or case-insensitive word: red, green, yellow, blue, magenta, cyan, or white)')

    return "\x1b[%s9%im" % (bold and "1;" or "", color) + text + "\x1b[0m"


# http://code.activestate.com/recipes/498245-lru-and-lfu-cache-decorators/
class Counter(dict):

    'Mapping where default values are zero'

    def __missing__(self, key):
        return 0


def lru_cache(maxsize=100):
    '''Least-recently-used cache decorator.

    Arguments to the cached function must be hashable.
    Cache performance statistics stored in f.hits and f.misses.
    Clear the cache with f.clear().
    http://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used

    '''
    maxqueue = maxsize * 10

    def decorating_function(user_function,
                            len=len, iter=iter, tuple=tuple, sorted=sorted, KeyError=KeyError):
        cache = {}                  # mapping of args to results
        queue = collections.deque()  # order that keys have been used
        refcount = Counter()        # times each key is in the queue
        sentinel = object()         # marker for looping around the queue
        kwd_mark = object()         # separate positional and keyword args

        # lookup optimizations (ugly but fast)
        queue_append, queue_popleft = queue.append, queue.popleft
        queue_appendleft, queue_pop = queue.appendleft, queue.pop

        @functools.wraps(user_function)
        def wrapper(*args, **kwds):
            # cache key records both positional and keyword args
            key = args
            if kwds:
                key += (kwd_mark,) + tuple(sorted(kwds.items()))

            # record recent use of this key
            queue_append(key)
            refcount[key] += 1

            # get cache entry or compute if not found
            try:
                result = cache[key]
                wrapper.hits += 1
            except KeyError:
                result = user_function(*args, **kwds)
                cache[key] = result
                wrapper.misses += 1

                # purge least recently used cache entry
                if len(cache) > maxsize:
                    key = queue_popleft()
                    refcount[key] -= 1
                    while refcount[key]:
                        key = queue_popleft()
                        refcount[key] -= 1
                    del cache[key], refcount[key]

            # periodically compact the queue by eliminating duplicate keys
            # while preserving order of most recent access
            if len(queue) > maxqueue:
                refcount.clear()
                queue_appendleft(sentinel)
                for key in ifilterfalse(refcount.__contains__,
                                        iter(queue_pop, sentinel)):
                    queue_appendleft(key)
                    refcount[key] = 1

            return result

        def clear():
            cache.clear()
            queue.clear()
            refcount.clear()
            wrapper.hits = wrapper.misses = 0

        wrapper.hits = wrapper.misses = 0
        wrapper.clear = clear
        return wrapper
    return decorating_function


def check_assignment_array_input(assignments, check_ndarray=True, check_integer=True, ndim=2):
    """Check if input is an appropriate data type for assignments.

    Parameters
    ----------
    assignments : ndarray
        Assignment data whose format will be checked.
    check_ndarray : bool, optional
        Default True; set False to skip checking for ndarray type
    check_integer : bool, optional
        Default True; set False to skip checking for integer dtype
    ndim : int, optional
        Default is 2, which is the correct value for an assignment array.

    Notes
    -----
    Checks if type is Numpy array, if dtype is int-like,
    and if ndim is ndim (2 by default).
    """

    if check_ndarray and not isinstance(assignments, np.ndarray):
        raise TypeError("Input assignments must be numpy array type.")

    if check_integer and assignments.dtype.kind != "i":
        raise TypeError("Input assignments must be integer type.")

    if assignments.ndim != ndim:
        raise TypeError("Input assignments must have ndim = %d; found %d." %
                        (ndim, assignments.ndim))
