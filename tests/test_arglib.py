import sys, os
from msmbuilder.testing import *
from msmbuilder.arglib import ArgumentParser, add_argument


def test_iterator():
    from msmbuilder.arglib import _iter_both_cases
    got = [e for e in _iter_both_cases("string")]
    expected = ['s', 'S', 't', 'T', 'r', 'R', 'i', 'I', 'n', 'N', 'g', 'G']

    eq_(got, expected)


class tester():
    def setup(self):
        self.p = ArgumentParser()
    
    def cmd(self, value):
        sys.argv = [os.path.abspath(__file__)]
        for e in value.split():
            sys.argv.append(e)
    
    @property
    def v(self):
        return self.p.parse_args(print_banner=False)

    def add(self, *args, **kwargs):
        self.p.add_argument(*args, **kwargs)

    def test_0(self):
        self.add('a', nargs='+')
        self.cmd('-a 1 2')
        eq_(self.v.a, ['1', '2'])
 
    def test_1(self):
        self.add('a', nargs='+', type=int)
        self.cmd('-a 1 2')
        
        eq_(self.v.a, [1, 2])
        
    def test_2(self):
        self.add('a', type=int)
        self.cmd('-a 4')
        eq_(self.v.a, 4)
    
    def test_3(self):
        self.add('a', type=float)
        self.cmd('-a 5')
        eq_(self.v.a, 5.0)
    
    @raises(SystemExit)
    def test_4(self):
        sys.stderr = open('/dev/null', 'w')
        self.add('a', choices=[1,2], type=float)
        self.cmd('-a 5')
        eq_(self.v.a, 5.0)
        sys.stderr = sys.__stderr__
    
    def test_5(self):
        self.add('a', choices=['1', '2'], type=int)
        self.cmd('-a 1')
        eq_(self.v.a, 1.0)
    
    def test_6(self):
        self.add('a', choices=[1, 2], type=int)
        self.cmd('-a 1')
        eq_(self.v.a, 1)
        
    def test_7(self):
        self.add('a', action='store_true', type=bool)
        self.cmd('-a')
        eq_(self.v.a, True)
    
    def test_71(self):
        self.add('a', action='store_false', type=bool)
        self.cmd('-a')
        eq_(self.v.a, False)
    
    def test_8(self):
        self.add('a', action='store_true')
        self.cmd('-a')
        eq_(self.v.a, True)
    
    def test_81(self):
        self.add('b', action='store_false')
        self.cmd('-b')
        eq_(self.v.b, False)

    def test_9(self):
        self.add('a', default=False, action='store_true', type=bool)
        self.cmd('')
        eq_(self.v.a, False)    

    def test_10(self):
        self.add('a', action='store_true', type=bool)
        self.cmd('')
        eq_(self.v.a, False)
