#!/usr/bin/env python
"""MSMBuilder: Analyzing conformational dynamics via the construction of Markov state models.
"""

##############################################################################
# Imports
##############################################################################

import re
import os
import sys
import inspect
import warnings
import argparse
from mdtraj.utils import import_
from msmbuilder import scripts

parser = argparse.ArgumentParser(description=__doc__, usage='msmb [subcommand]')

##############################################################################
# Code
##############################################################################

def entry_point():
    subparsers = parser.add_subparsers(dest="subparser_name")
    scriptfiles = {}
    argv = sys.argv[:]
    if len(argv) == 1:
        argv.append('-h')

    for scriptname in scripts.__all__:
        # get the name and first sentence of the description from each of the
        # msmbuilder commands
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            script = import_('msmbuilder.scripts.%s' % scriptname)
            scriptparser = getattr(script, 'parser', None)
        scriptfiles[scriptname] = script.__file__

        try:
            description = scriptparser.description
        except:
            description = scriptparser.parser.description

        # http://stackoverflow.com/a/17124446/1079728
        first_sentence = ' '.join(' '.join(re.split(r'(?<=[.:;])\s', description)[:1]).split())
        subparsers.add_parser(scriptname, help=first_sentence)

    args = parser.parse_args(argv[1:2])        
    sys.argv = argv[1:]
    getattr(scripts, args.subparser_name).entry_point()
        
if __name__ == '__main__':
    entry_point()
