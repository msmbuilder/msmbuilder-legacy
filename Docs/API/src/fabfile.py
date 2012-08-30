"""
Simple script to deploy the sphinx documentation
to http://stanford.edu/~rmcgibbo/msmbuilder

>> fab -H corn.stanford.edu deploy

"""

from fabric.api import run
import os

def deploy():
    run('cd ~/msmbuilder/msmbuilder; svn update')
    run('cd ~/msmbuilder/sandbox/rmcgibbo/sphinx; svn update')
    run('cd ~/msmbuilder/msmbuilder; python setup.py install')
    run('cd ~/msmbuilder/sandbox/rmcgibbo/sphinx; make clean; rm -rf generated; make html')
    run('rm -rf ~/WWW/msmbuilder/')
    run('cp -r ~/msmbuilder/sandbox/rmcgibbo/sphinx/_build/html ~/WWW/msmbuilder')
    