import sys, os
import argparse
import numpy as np
import functools
from msmbuilder import Serializer
from msmbuilder import Project
from msmbuilder import Trajectory
from msmbuilder.License import LicenseString
from msmbuilder.Citation import CiteString
import scipy.io
from collections import namedtuple

def _iter_both_cases(string):
    for c in string:
        yield c
        yield c.swapcase()

def _LoadType(load_function):
    def F(path):
        if not os.path.exists(path):
            raise IOError('Could not find file %s' % path)
        try:
            return load_function(path)
        except:
            raise IOError('Could not open %s' % path)
    return F
    
ProjectType = _LoadType(Project.LoadFromHDF)
SerializerType = _LoadType(Serializer.LoadFromHDF)
TrajectoryType = _LoadType(Trajectory.LoadTrajectoryFile)

def LoadTxtType(**kwargs):
    return functools.partial(np.loadtxt, **kwargs)
    
def die_if_path_exists(path):
    if isinstance(path, list):
        map(die_if_path_exists, path)
        return None
    
    directory = os.path.split(path)[0]
    if len(directory) > 0 and not os.path.exists(directory):
        print 'Creating directory %s' % directory
        os.makedirs(directory)
    if os.path.exists(path):
        name = os.path.split(sys.argv[0])[1]
        print >> sys.stderr, '%s: Error: %s already exists!. Exiting.' % (name, path)
        sys.exit(1)
    
    return None

def ensure_path_exists(path):
    name = os.path.split(sys.argv[0])[1]
    if not os.path.exists(path):
        print >> sys.stderr, "%s: Error: Can't find %s" % (name, path)
    
RESERVED = {'assignments': ('-a', 'Path to assignments file.', 'Data/Assignments.h5', SerializerType),
            'project': ('-p', 'Path to ProjectInfo file.', 'ProjectInfo.h5', ProjectType),
            'tProb': ('-t', 'Path to transition matrix.', 'Data/tProb.mtx', scipy.io.mmread),
            'output_dir': ('-o', 'Location to save results.', 'Data/', str),
            'pdb': ('-s', 'Path to PDB structure file.', None, str)}

nestedtype = namedtuple('nestedtype', 'innertype')

def add_argument(group, name, description=None, type=None, choices=None, nargs=None, default=None, action=None):
    if name in RESERVED:
        short = RESERVED[name][0]
        if description is None:
            description = RESERVED[name][1]
        if default is None:
            default = RESERVED[name][2]
        if type is None:
            type = RESERVED[name][3]

    if type is None and nargs is None:
        type = str
    if nargs is not None:
        if type is None:
            type = nestedtype(str)
        else:
            type = nestedtype(type)        
    
    kwargs = {}
    
    if action is not None:
        kwargs['action'] = action
    if nargs is not None:
        kwargs['nargs'] = nargs
    if choices is not None:
        kwargs['choices'] = choices

    long = '--{name}'.format(name=name)
    found_short = False
    
    for char in _iter_both_cases(name):
        if not name in RESERVED:
            short = '-%s' % char

        args = (short, long)

        if default is None:
            kwargs['required'] = True
            kwargs['help'] = description
        else:
            if description is None:
                helptext = 'Default: {default}'.format(default=default)
            else:
                if description[-1] != '.':
                    description += '.'
                helptext = '{description} Default: {default}'.format(description=description, default=default)
            kwargs['help'] = helptext
            kwargs['default'] = default
        try:
            group.add_argument(*args, **kwargs)
            found_short = True
            break
        except argparse.ArgumentError as e:
            pass
    if not found_short:
        raise ValueError('Could not find short name')
    
    return name, type

class ArgumentParser(object):
    "MSMBuilder specific wrapper around argparse.ArgumentParser"
    
    def __init__(self, *args, **kwargs):
        """Create an ArgumentParser
        
        Parameters
        ----------
        description: (str, optional)
        
        
        """
        
        if 'description' in kwargs:
            kwargs['description'] += ('\n' + '-'*80)
        kwargs['formatter_class'] = argparse.RawDescriptionHelpFormatter
        
        self.parser = argparse.ArgumentParser(self, *args, **kwargs)
        self.parser.prog=os.path.split(sys.argv[0])[1]
        self.required = self.parser.add_argument_group(title='required arguments')
        self.wdefaults = self.parser.add_argument_group(title='arguments with defaults')
        
        self.short_strings = set(['-h'])
        self.name_to_type = {}
        
        for v in RESERVED.values():
            self.short_strings.add(v[0])
            
        self.extra_groups = []
    
    def add_argument_group(self, title):
        self.extra_groups.append(self.parser.add_argument_group(title=title))
    
    def add_argument(self, name, description=None, type=None, choices=None, nargs=None, default=None, action=None):
        if name in RESERVED and default is None:
            default = RESERVED[name][2]

        if default is None:
            group = self.required
        else:
            group = self.wdefaults

        if len(self.extra_groups) > 0:
            group = self.extra_groups[-1]

        name, type = add_argument(group, name, description, type, choices, nargs, default, action)

        self.name_to_type[name] = type
     
    def parse_args(self):
        print LicenseString
        print CiteString
        namespace = self.parser.parse_args()
        return self._typecast(namespace)
    
    def _typecast(self, namespace):
        """Work around for the argparse bug with respect to defaults and FileType not
        playing together nicely -- http://stackoverflow.com/questions/8236954/specifying-default-filenames-with-argparse-but-not-opening-them-on-help"""
        for name, type in self.name_to_type.iteritems():
            if isinstance(type, nestedtype):
                setattr(namespace, name, [type.innertype(e) for e in getattr(namespace, name)])
            else:
                setattr(namespace, name, type(getattr(namespace, name)))
                
        return namespace
        
if __name__ == '__main__':
    a = ArgumentParser(description='myparser')
    a.add_argument('project', choices=['a', 'b'])
    args = a.parse_args()
    print args
