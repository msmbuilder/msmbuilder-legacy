# This file is part of MSMBuilder.
#
# Copyright 2011 Stanford University
#
# MSMBuilder is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

"""Tools for constructing MSMBuilder projects from directories containing xtcs.
"""
import os
from numpy import array,argmax
import string

from msmbuilder import utils
from msmbuilder import Trajectory, Conformation, Project
from msmbuilder.metrics import RMSD
import multiprocessing
try:
    from deap import dtm
except:
    pass
import logging
logger = logging.getLogger('CreateMergedTraj')

def CreateMergedTrajectoriesFromFAH(PDBFilename,DataDir,NumRuns,NumClones,InFilenameRoot="frame",OutFilenameRoot="trj",OutDir="./Trajectories",OutFileType=".lh5",WhichRunsClones=None,Stride=1,NamingConvention=0,AtomIndices=None,Usetrjcat=False,MaxRMSD=7,DiscardHighRMSD=True,MaxGen=100000,MinGen=0,DiscardFirstN=0,trjcatFlags=["-cat"],ProjectFilename="ProjectInfo.ph5",CenterConformations=True,SkipTrajCat=False):
    """Create an MSMBuilder Project from a FAH project, making fragmented trajectories (generations) whole.

    Inputs:
    PDBFilename -- a PDB file with correct atom names and indexing.
    DataDir -- the location of the data to be converted.
    NumRuns -- the number of runs in the FAH project.
    NumClones -- the number of clones in the FAH project.

    Keyword Arguments:
    InFilenameRoot -- The 'root' name of individual input trajectory files.  Default: 'frame'
    OutFilenameRoot -- The 'root' name for output trajectory files.  Default: 'trj'
    OutDir -- Place output Trajectories here.  Default: './Trajectories'
    OutFileType -- The file type for storing output trajectories.  Default: '.lh5'
    WhichRunsClones -- An optional list of form [[run_i, clone_i],[run_j, clone_j] ...] to specify runs,clones.  Default: None
    Stride -- Subsample input data by every Nth frame.  Default: 1
    NamingConvention -- Specify whether input data numbering looks like 000, 001, 002 or 0, 1, 2.  Default: 0
    AtomIndices -- A numpy array of atom indices to include in ouput data.
    While this can be used to strip waters, it is generally faster to use Gromacs to pre-process XTCs.  Default: None
    Usetrjcat -- Use Gromacs trjcat to concatenate trajectories, rather than the msmbuilder xtc library.  Default: False
    MaxRMSD -- Calculate RMSD to PDBFilename and reject data that is larger than MaxRMSD.  Useful to test for 'blowing up'.  Default: 7 [nm]
    DiscardHighRMSD -- Calculate RMSD to PDBFilename and reject data that is larger than MaxRMSD.  Useful to test for 'blowing up'.  Default: True
    MaxGen -- Discard data after MaxGen generations.  Default: 100000
    MinGen -- Discard trajectories that contain fewer than MinGen generations.  Default: 0
    DiscardFirstN -- Discard the first N generations for each RUNCLONE.  Default: 0
    trjcatFlags -- Used to pass command line arguments to Gromacs trjcat.  Default: ['-cat']
    ProjectFileName -- Filename of Project to output.  Default: 'ProjectInfo.ph5'
    CenterConformations -- Center conformations before saving.  Default: True
    SkipTrajCat -- Skip TrajCat step.  Default: False
    """
    try:
        os.mkdir(OutDir)
    except OSError:
        logger.warning("The directory %s already exists." % OutDir)
    RunList=[]
    CloneList=[]
    NumGensList=[]
    Conf1=Conformation.Conformation.LoadFromPDB(PDBFilename)
    if WhichRunsClones!=None:
        WhichRunsClones=WhichRunsClones.tolist()

    TrajNumber=array(0)#Use array because it passes by reference, rather than value.
    for Run in range(NumRuns):
        for Clone in range(NumClones):
            ConvertRunClone(Run,Clone,TrajNumber,Conf1,DataDir,RunList,CloneList,NumGensList,Usetrjcat=Usetrjcat,trjcatFlags=trjcatFlags,DiscardFirstN=DiscardFirstN,Stride=Stride,WhichRunsClones=WhichRunsClones,OutDir=OutDir,InFilenameRoot=InFilenameRoot,OutFilenameRoot=OutFilenameRoot,OutFileType=OutFileType,MinGen=MinGen,MaxGen=MaxGen,AtomIndices=AtomIndices,DiscardHighRMSD=DiscardHighRMSD,MaxRMSD=MaxRMSD,PDBFilename=PDBFilename,CenterConformations=CenterConformations,SkipTrajCat=SkipTrajCat, NamingConvention=NamingConvention)
            
    logger.info("Creating Project File")
    P1=Project.CreateProjectFromDir(ConfFilename=PDBFilename,TrajFileType=OutFileType,RunList=RunList,CloneList=CloneList,NumGensList=NumGensList,Filename=ProjectFilename)
    return(P1)

def GetFilename(Run,Clone,InFilenameRoot,Gen,DataDir,NamingConvention=0):
    """Convert Run Clone Gen to filename.  If your files are named differently, create a new NamingConvention case."""
    if NamingConvention==1:
        Filename="%s/RUN%d/CLONE%d/%s%.3d.xtc"%(DataDir,Run,Clone,InFilenameRoot,Gen)
    elif NamingConvention==0:
        Filename="%s/RUN%d/CLONE%d/%s%d.xtc"%(DataDir,Run,Clone,InFilenameRoot,Gen)
    elif NamingConvention==2:
        Filename="%s/RUN%d/CLONE%d/%s%d.trr"%(DataDir,Run,Clone,InFilenameRoot,Gen)
    return(Filename)

def DetermineNumGens(Run,Clone,InFilenameRoot,FilenameList,DataDir,NamingConvention=0):
    """Find the maximal n such that the following files exist: frame0.xtc, ... , frame(n-1).xtc."""
    Gen=0
    KeepGoing=True
    while KeepGoing:
        Filename=GetFilename(Run,Clone,InFilenameRoot,Gen,DataDir,NamingConvention=NamingConvention)
        if not os.path.exists(Filename):
            KeepGoing=False
        else:
            FilenameList.append(Filename)
            Gen=Gen+1
    return(len(FilenameList))

def ConvertRunClone(Run,Clone,TrajNumber,Conf1,DataDir,RunList,CloneList,NumGensList,Usetrjcat=False,trjcatFlags=["-cat"],DiscardFirstN=0,Stride=1,WhichRunsClones=None,NamingConvention=0,OutDir="./Trajectories/",InFilenameRoot="frame",OutFilenameRoot="trj",OutFileType=".lh5",MinGen=0,MaxGen=100000,AtomIndices=None,DiscardHighRMSD=True,MaxRMSD=7,PDBFilename=None,CenterConformations=True,SkipTrajCat=False):
    """Convert a single Run Clone into msmbuilder format (e.g. .lh5).  

    Inputs:
    Run -- Which (input) Run.
    Clone -- Which (input) Clone.
    TrajNumber -- The number of the current ouput Trajectory.
    Conf1 -- A Conformation object with correct atom names and indexing.  Necessary to load XTCs.
    DataDir -- location of input Data
    RunList -- keep track of the current run so msmbuilder project knows where output trajectory came from.
    CloneList -- keep track of the current clone so msmbuilder project knows where output trajectory came from.
    NumGensList -- How many Generators for current run clone.

    Keyword Arguments:
    PDBFilename -- filename of PDB with correct atom names and indexing.  
    InFilenameRoot -- The 'root' name of individual input trajectory files.  Default: 'frame'
    OutFilenameRoot -- The 'root' name for output trajectory files.  Default: 'trj'
    OutDir -- Place output Trajectories here.  Default: './Trajectories'
    OutFileType -- The file type for storing output trajectories.  Default: '.lh5'
    WhichRunsClones -- An optional list of form [[run_i, clone_i],[run_j, clone_j] ...] to specify runs,clones.  Default: None
    Stride -- Subsample input data by every Nth frame.  Default: 1
    NamingConvention -- Specify whether input data numbering looks like 000, 001, 002 or 0, 1, 2.  Default: 0
    AtomIndices -- A numpy array of atom indices to include in ouput data.
    While this can be used to strip waters, it is generally faster to use Gromacs to pre-process XTCs.  Default: None
    Usetrjcat -- Use Gromacs trjcat to concatenate trajectories, rather than the msmbuilder xtc library.  Default: False
    MaxRMSD -- Calculate RMSD to PDBFilename and reject data that is larger than MaxRMSD.  Useful to test for 'blowing up'.  Default: 7 [nm]
    DiscardHighRMSD -- Calculate RMSD to PDBFilename and reject data that is larger than MaxRMSD.  Useful to test for 'blowing up'.  Default: True
    MaxGen -- Discard data after MaxGen generations.  Default: 100000
    MinGen -- Discard trajectories that contain fewer than MinGen generations.  Default: 0
    DiscardFirstN -- Discard the first N generations for each RUNCLONE.  Default: 0
    trjcatFlags -- Used to pass command line arguments to Gromacs trjcat.  Default: ['-cat']
    CenterConformations -- Center conformations before saving.  Default: True
    SkipTrajCat -- Skip TrajCat step.  Default: False
    """
    
    pconf1 = pconf1 = RMSD(AtomIndices).prepare_trajectory({'XYZList': array([Conf1['XYZ']])})
    
    
    logger.info("RUN%d CLONE%d", Run, Clone)
    if WhichRunsClones!=None:
        if [Run,Clone] not in WhichRunsClones:
            logger.warning("RUN%d CLONE%d was not selected for inclusion in this project; skipping.", Run, Clone)
            return
    FilenameList=[]
    NumGens=DetermineNumGens(Run,Clone,InFilenameRoot,FilenameList,DataDir,NamingConvention=NamingConvention)
    NumGens=min(MaxGen,NumGens)
    if NumGens<MinGen or NumGens<=0:
        logger.warning("Skipping Run %d Clone %d; too few generations (%d) available.", Run,Clone,NumGens)
        return
    FilenameList=FilenameList[0:NumGens]
    OutFilename="%s/%s%d%s"%(OutDir,OutFilenameRoot,TrajNumber,OutFileType)
    if not os.path.exists(OutFilename):
        if SkipTrajCat==False:
            if Usetrjcat==False:
                try: Traj=Trajectory.LoadFromXTC(FilenameList,Conf=Conf1)  # Try to skip bad trajs
                except IOError:
                    corrupted_xtcs = True
                    xtc_ind = 1
                    while corrupted_xtcs:
                        logger.warning("Found corrupted XTC: %s", FilenameList[-xtc_ind])
                        logger.warning("Attempting to recover by discarding this %d-to-last frame...", xtc_ind)
                        try:
                            Traj=Trajectory.LoadFromXTC(FilenameList[:-xtc_ind], Conf=Conf1)
                        except IOError:
                            xtc_ind += 1 
                        else:
                            corrupted_xtcs = False
            else:
                CMD="trjcat -f %s %s"%(string.join(FilenameList),string.join(trjcatFlags))
                os.system(CMD)
                Traj=Trajectory.LoadFromXTC("trajout.xtc",PDBFilename=PDBFilename)
                os.remove("trajout.xtc")
            Traj["XYZList"]=Traj["XYZList"][DiscardFirstN::Stride]
            if AtomIndices!=None:
                Traj.RestrictAtomIndices(AtomIndices)
                #Traj["XYZList"]=Traj["XYZList"][:,AtomIndices,:]
            if DiscardHighRMSD==True:
                #rmsd=Traj.CalcRMSD(Conf1)
                ptraj = RMSD().prepare_trajectory(Traj)
                rmsd = RMSD().one_to_all(pconf1, ptraj, 0)
                if max(rmsd)>MaxRMSD:
                    logger.warning("Frame %d has RMSD %f and appears to be blowing up or damaged.  Dropping Trajectory.", argmax(rmsd), max(rmsd))
                    return   
            if CenterConformations==True:
                RMSD.TheoData.centerConformations(Traj["XYZList"])
            Traj.Save(OutFilename)
        else:
            CMD="trjcat -f %s %s"%(string.join(FilenameList),string.join(trjcatFlags))
            os.system(CMD)
            Traj=Trajectory.LoadFromXTC("trajout.xtc",PDBFilename=PDBFilename)
            os.remove("trajout.xtc")
    else:
        logger.warning("Already Found File %s; skipping", OutFilename)
    RunList.append(Run)
    CloneList.append(Clone)
    NumGensList.append(NumGens)
    TrajNumber+=1

def CreateMergedTrajectories(PDBFilename,ListOfXTCList,OutFilenameRoot="trj",
                             OutDir="./Trajectories",OutFileType=".lh5",Stride=1,
                             AtomIndices=None,InFileType=".xtc", parallel=None):
    """Create Merged Trajectories from a list of xtcs.

    Inputs:
    PDBFilename -- Filename of PDB with correct atom names and indexing.
    ListOfXTCList -- A list of lists containing XTC files for each trajectory.  Each inner list will be merged into 'whole' trajectories.

    Keyword Arguments:
    OutFilenameRoot -- The 'root' file name for output trajectories.  Default: 'trj'
    OutDir -- The directory to store output trajectories.  Default: './Trajectories'
    OutFileType -- Format of output files.  Default: '.lh5'
    InFileType -- Format of input files.  Default: '.xtc'
    Stride -- Subsample input data by every Nth frame.  Default: 1
    AtomIndices -- Only store specific atoms from the input data.  Default: None 
    """
    try:
        os.mkdir(OutDir)
    except OSError:
        logger.error("The directory %s already exists.  Exiting!", OutDir)
        return
    

    if parallel == 'multiprocessing':
        pool = multiprocessing.Pool()
        mymap = pool.map
    elif parallel == 'dtm':
        mymap = dtm.map
    else:
        mymap = map
        
    mymap(_ConvertFilenameList, utils.uneven_zip(ListOfXTCList, range(len(ListOfXTCList)),
        [PDBFilename], [InFileType], [OutDir], [OutFilenameRoot], [OutFileType], [Stride], [AtomIndices]))
    
def _ConvertFilenameList(args):
        FilenameList, i, PDBFilename, InFileType, OutDir, OutFilenameRoot, OutFileType, Stride, AtomIndices = args
        #FilenameList=ListOfXTCList[i]
        if len(FilenameList)>0:
            logger.info(FilenameList)
            if InFileType =='.dcd': # TG should auto-switch instead
                Traj=Trajectory.LoadFromDCD(FilenameList,PDBFilename=PDBFilename)
            else:
                Traj=Trajectory.LoadFromXTC(FilenameList,PDBFilename=PDBFilename)
            Traj["XYZList"]=Traj["XYZList"][::Stride]
            if AtomIndices!=None:
                Traj.RestrictAtomIndices(AtomIndices)
                #Traj["XYZList"]=Traj["XYZList"][:,AtomIndices,:].copy()
            Traj.Save("%s/%s%d%s"%(OutDir,OutFilenameRoot,i,OutFileType))
