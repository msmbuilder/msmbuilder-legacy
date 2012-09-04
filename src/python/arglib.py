import sys, os
import argparse
from msmbuilder.License import LicenseString
from msmbuilder.Citation import CiteString
from msmbuilder import metric_parsers

import warnings

def _iter_both_cases(string):
    """Iterate over the chars in a strings in both cases
    
    >>> [e for e in _iter_both_cases("string")]
    ['s', 'S', 't', 'T', 'r', 'R', 'i', 'I', 'n', 'N', 'g', 'G']    
    """
    for c in string:
        yield c
        yield c.swapcase()

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
    
RESERVED = {'assignments': ('-a', 'Path to assignments file.', 'Data/Assignments.h5', str),
            'project': ('-p', 'Path to ProjectInfo file.', 'ProjectInfo.h5', str),
            'tProb': ('-t', 'Path to transition matrix.', 'Data/tProb.mtx', str),
            'output_dir': ('-o', 'Location to save results.', 'Data/', str),
            'pdb': ('-s', 'Path to PDB structure file.', None, str),}

def add_argument(group, name, description=None, type=None, choices=None, nargs=None, default=None, action=None):
    """
    Wrapper around arglib.ArgumentParser.add_argument. Gives you a short name
    directly from your longname
    """
    if name in RESERVED:
        short = RESERVED[name][0]
        if description is None:
            description = RESERVED[name][1]
        if default is None:
            default = RESERVED[name][2]
        if type is None:
            type = RESERVED[name][3]

    kwargs = {}
    if action == 'store_true' or action == 'store_false':
        type = bool
    if type != None:
        kwargs['type']=type
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
        except argparse.ArgumentError:
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
        get_metric: (bool, optional) - Pass true if you want to use the metric parser and get a metric instance returned
        
        """

        self.print_argparse_bug_warning = False        

        if 'description' in kwargs:
            kwargs['description'] += ('\n' + '-'*80)
        kwargs['formatter_class'] = argparse.RawDescriptionHelpFormatter
        
        if 'get_metric' in kwargs:
            self.get_metric = bool(kwargs.pop('get_metric')) # pop gets the value plus removes the entry

        self.parser = argparse.ArgumentParser(*args, **kwargs)

        self.parser.add_argument('-q','--quiet',dest='quiet',help='Pass this flag to run in quiet mode.',default=False,action='store_true')

        if self.get_metric:
            metric_parsers.add_metric_parsers( self )

        self.parser.prog = os.path.split(sys.argv[0])[1]
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

        if action == 'store_true':
            default = False
        elif action == 'store_false':
            default = True

        if (nargs in ['+','*','?']) and ( self.get_metric ):
            self.print_argparse_bug_warning = True

        if choices:    
            for choice in choices:
                if not isinstance(choice, str):
                    warnings.warn('arglib bug: choices should all be str')

        if default is None:
            group = self.required
        else:
            group = self.wdefaults

        if len(self.extra_groups) > 0:
            group = self.extra_groups[-1]

        name, type = add_argument(group, name, description, type, choices, nargs, default, action)

        self.name_to_type[name] = type
     
    def add_subparsers(self, *args, **kwargs):
        return self.parser.add_subparsers(*args,**kwargs)

    def parse_args(self, args=None, namespace=None, print_banner=True):
        if print_banner:
            print LicenseString
            print CiteString

        if self.print_argparse_bug_warning:
            print "#"*80
            print "\n"
            warnings.warn('Known bug in argparse regarding subparsers and optional arguments with nargs=[+*?] (http://bugs.python.org/issue9571)')
            print "\n"
            print "#"*80

        namespace = self.parser.parse_args(args=args, namespace=namespace)

        print namespace
        #namespace = self._typecast(namespace)
        if self.get_metric: # if we want to get the metric, then we have to construct it
            metric = metric_parsers.construct_metric( namespace )
            return namespace, metric

        return namespace
    
    def _typecast(self, namespace):
        """Work around for the argparse bug with respect to defaults and FileType not
        playing together nicely -- http://stackoverflow.com/questions/8236954/specifying-default-filenames-with-argparse-but-not-opening-them-on-help"""
        for name, type in self.name_to_type.iteritems():
            if isinstance(type, nestedtype):
                setattr(namespace, name, [type.innertype(e) for e in getattr(namespace, name)])
            else:
                setattr(namespace, name, type(getattr(namespace, name)))
                
        return namespace
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
