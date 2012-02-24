from scipy import signal
import numpy as np
import re

def fft_acf(A):
    '''Return the autocorrelation of a 1D array using the fft
    Note: the result is normalized'''
    A = A - np.mean(A)
    result = signal.fftconvolve(A, A[::-1])
    result = result[result.size / 2:] 
    return result / result[0]

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
                result[j] = args[j][lengths[j]-1]
        return result
    zipped = map(get, range(max(lengths)))
    return zipped



def format_block(block):
    '''Format the given block of text, trimming leading/trailing
    empty lines and any leading whitespace that is common to all lines.
    The purpose is to let us list a code block as a multiline,
    triple-quoted Python string, taking care of indentation concerns.'''
    # separate block into lines
    lines = str(block).split('\n')
    # remove leading/trailing empty lines
    while lines and not lines[0]:  del lines[0]
    while lines and not lines[-1]: del lines[-1]
    # look at first line to see how much indentation to trim
    ws = re.match(r'\s*',lines[0]).group(0)
    if ws:
            lines = map( lambda x: x.replace(ws,'',1), lines )
    # remove leading/trailing blank lines (after leading ws removal)
    # we do this again in case there were pure-whitespace lines
    while lines and not lines[0]:  del lines[0]
    while lines and not lines[-1]: del lines[-1]
    return '\n'.join(lines)+'\n'

#if __name__ == '__main__':
#    print uneven_zip([1,2,3], 'a', [10,11])
## {{{ http://code.activestate.com/recipes/576862/ (r1)
"""
doc_inherit decorator

Usage:

class Foo(object):
    def foo(self):
        "Frobber"
        pass

class Bar(Foo):
    @doc_inherit
    def foo(self):
        pass 

Now, Bar.foo.__doc__ == Bar().foo.__doc__ == Foo.foo.__doc__ == "Frobber"
"""

from functools import wraps

class DocInherit(object):
    """
    Docstring inheriting method descriptor

    The class itself is also used as a decorator
    """

    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):

        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=('__name__','__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):

        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden: break

        @wraps(self.mthd, assigned=('__name__','__module__'))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            raise NameError, ("Can't find '%s' in parents"%self.name)
        func.__doc__ = source.__doc__
        return func

doc_inherit = DocInherit 
## end of http://code.activestate.com/recipes/576862/ }}}
