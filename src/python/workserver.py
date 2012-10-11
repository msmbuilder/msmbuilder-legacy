"""
This is a class that contains all of the necessary tools to interact
with folding@home projects.

Written by: TJ Lane <tjlane@stanford.edu>
Contributions from Robert McGibbon
"""

# GLOBAL IMPORTS
import os
import re
import sys
import cPickle
import time
from glob import glob

import smtplib
from email.mime.text import MIMEText

from numpy import argmax

import subprocess
from subprocess import PIPE

from multiprocessing import Pool
try:
    from deap import dtm
except:
    pass

from msmbuilder import Trajectory
from msmbuilder.metrics import RMSD
from msmbuilder import Project
from msmbuilder.utils import make_methods_pickable, keynat
make_methods_pickable()
import logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger('FahProject')


class FahProject(object):
    """
    A generic class for interacting with Folding@home projects

    Parameters
    ----------
    pdb : str
        The pdb file on disk associated with the project.

    project_number: int
        The project number assocaited with the project.

    projectinfo_file : str
        Name of the project info file.

    work_server : str
        Hostname of the work server to interact with.

    email : str
        email to forward alerts to
    """

    def __init__(self, pdb, project_number=0001, projectinfo_file="ProjectInfo.h5", 
                 work_server=None, email=None):

        # metadata associated with a FAH project
        self.project_number   = project_number
        self.work_server      = work_server
        self.pdb_topology     = pdb
        self.manager_email    = email
        self.projectinfo_file = projectinfo_file

		# check that the PDB exists
        if not os.path.exists(self.pdb):
            logger.error("Cannot find %s", self.pdb)


    def restart_server(self):
        """
        Restarts the workserver, should be called when injecting runs.

        Checks that the server comes back up without throwing an error -
        if it doesn't come up OK, sends mail to the project manager.
        """

        raise NotImplementedError()

        # restart the server, wait 60s to let it come back up
        logger.warning("Restarting server: %s", self.work_server)
        stop_cmd  = "/etc/init.d/FAHWorkServer-%s stop" % self.work_server
        start_cmd = "/etc/init.d/FAHWorkServer-%s start" % self.work_server
        r = subprocess.call(stop_cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        time.sleep(60)
        r = subprocess.call(stop_cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)

        # check that we came back up OK, if not freak out
        processname = "FAHWorkServer-%s" % self.work_server
        is_alive = False # guilty until proven innocent

        for line in os.popen("ps -a"):
            if line.find(processname) > 0:
                is_alive = True

        if not is_alive:
            error_msg = """
                    FATAL ERROR: msmbuilder.workserver is reporting a critical issue:

                             Workserver %s did not come back up after restart!

                    Recommend you attend to this immediately.""" % self.workserver

            if email: send_error_email(self, error_msg)
            raise Exception(error_msg)

        return



    def send_error_email(self, error_msg):
        """
        Sends an error message to the registered email.

        Parameters
        ----------
        error_msg : str
            The string to include in the email.
        """

        raise NotImplementedError()

        if email == None:
            logger.error("Cannot send error email - no email provided")
            return

        msg = MIMEText(error_msg)

        msg['Subject'] = '[msmbuilder.FahProject] FATAL ERROR IN FAHPROJECT'
        msg['From'] = 'msmbuilder@gmail.com'
        msg['To'] = self.email

        # Send the message via our own SMTP server, but don't include the envelope header.
        logger.error("Sending error email to: %s", self.email)
        s = smtplib.SMTP('smtp.gmail.com')
        s.sendmail(me, [you], msg.as_string())
        s.quit()

        return


    def set_project_basepath(self):
        """
        Finds and internally stores a FAH Project's path.
        """
        search = glob("/home/*/server2/data/SVR*/PROJ%d" % self.project_number)
        if len(search) != 1:
            raise Exception("Could not find unique FAH project: %d on %s" % (self.project_number,
                                                                             self.work_server))
        else: self.project_basepath = search[0]


    def new_run(self):
        """
        Creates a new run in the project directory, and adds that run
        to the project.xml file. Does not directly reboot the server.
        """

        # create the new run directory
        raise NotImplementedError()
        # add the run to the project.xml


    def stop_run(self, run):
        """
        Stops all CLONES in a RUN.

        Parameters
        ----------
        run : int
            The run to stop.
        """

        logger.warning("Shutting down RUN%d", run)
        clone_dirs = glob(run_dir + "CLONE*")
        for clone_dir in clone_dirs:
            g = re.search('CLONE(\d+)', 'CLONE55')
            if g:
                clone = g.group(1)
                self.stop_clone(run, clone)

        return


    def stop_clone(self, run, clone):
        """
        Stops the specified RUN/CLONE by changing the name of
        the WU's trr, adding .STOP to the end.

        Parameters
        ----------
        run : int
            The run containing the clone to stop.

        clone : int
            The clone to stop.
        """

        clone_dir = os.path.join(self.project_basepath, 'RUN%d/' % run, 'CLONE%d/' % clone)

        # add .STOP to all dem TRR files
        trrs = glob( clone_dir + '*.trr' )
        if len(trrs) == 0:
            logger.error("Could not find any TRRs to stop in %s. Proceeding.", clone_dir)
        else:
            for trr in trrs:
                os.rename(trr, trr+'.STOP')
                loggger.info("Stopped: %s", trr)

        return




