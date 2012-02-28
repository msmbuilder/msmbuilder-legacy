"""
NOTE: This file should be run inside a pylab (ipython -pylab) environment, as I assume that the plotting functions are in the global namespace.  To run a script while in ipython -pylab, use
%run -i BuildMSMFromScratch.py

0.  Convert your data into XTC or HDF5 files using CreateTrajsFromFAH.py or CreateTrajsFromList.py (or use the data at TestFiles/ExampleProject).
1.  Load the current Project.  Trajectory data is typically saved as ./Trajectories/trj0.h5 ... ./Trajectories/trjN.h5.  
2.  Cluster the Project.  
3.  Assign the project, saving the Assignments and RMSD to cluster centers in the ./Data directory.  
4.  
5.  Calculate and plot implied timescales.
6.  Construct a Count Matrix and Transition matrix at the desired LagTime.  
7.  Sample Random Conformations from each state.
8.  Plot Free Energy Vs. RMSD
"""
from msmbuilder import Project,Conformation,Trajectory,MSMLib,Serializer
import scipy.io

ProjectFilename="ProjectInfo.h5"
PDBFilename="native.pdb"

#Step 1
P1=Project.Project.LoadFromHDF(ProjectFilename)
C1=Conformation.Conformation.LoadFromPDB(PDBFilename)
NumStates=100
a=C1["AtomNames"]
AInd=where((a=="N")|(a=="C")|(a=="CA")|(a=="O")|(a=="CB"))[0]

#Step 2.
Generators=P1.ClusterProject(NumStates,Stride=1,AtomIndices=AInd)
Generators.SaveToLHDF("Data/Gens.lh5")
#If you want to view the generators, save to PDB or XTC.  Otherwise, save to lossy HDF5 (lh5).  
#(Generators.SaveToPDB("Data/Gens.pdb"),Generators.SaveToHDF("Data/Gens.h5"), Generators.SaveToLHDF("Data/Gens.lh5"),Generators.SaveToXTC("Data/Gens.xtc"))
#Note that generators is a Trajectory object


#Step 3. 
Assignments,RMSD,WhichTrajsWereAssigned=P1.AssignProject(Generators)
Serializer.SaveData("Data/Assignments.h5",Assignments)
Serializer.SaveData("Data/Assignments.h5.RMSD",RMSD)

#Load your data; this is redundant as it's already loaded, but I have to show you how.
Assignments=Serializer.LoadData("Data/Assignments.h5")
RMSD=Serializer.LoadData("Data/Assignments.h5.RMSD")

#Step 4.  

#Construct a count matrix with shortest possible lagtime (1)
Counts=MSMLib.GetCountMatrixFromAssignments(Assignments,NumStates,LagTime=1,Slide=True)

#Calculate the state that has the most counts
X0=array((Counts+Counts.transpose()).sum(0)).flatten()
X0=X0/sum(X0)
MaxState=argmax(X0)

#Starting at the state with maximal number of counts, find the maximal connected subgraph.
#Discard all data that is disconnected from this subgraph.
#Note that the state with maximal counts is just a heuristic for finding the 'native' state and is known to fail occasionally.
#Note that IterativeTrim changes the state indexing by removing states!!!!
DesiredLagTime=10
MSMLib.IterativeTrim(Assignments,DesiredLagTime,Symmetrize=True,Start=MaxState)
Serializer.SaveData("Data/Assignments.Fixed.h5",Assignments)
NumStates=max(Assignments.flatten())+1


#Step 5.
NumEigen=6
for Time in arange(1,DesiredLagTime):
	Counts=MSMLib.GetCountMatrixFromAssignments(Assignments,NumStates,LagTime=Time,Slide=True)
	T=MSMLib.EstimateTransitionMatrix(Counts+Counts.transpose())
	EigAns=MSMLib.GetEigenvectors(T,NumEigen);
	plot(Time*ones(NumEigen),-Time/log(EigAns[0]),'o')


title("Implied Timescales Versus Lagtime")
yscale('log')

#Step 6
Time=DesiredLagTime
Counts=MSMLib.GetCountMatrixFromAssignments(Assignments,NumStates,LagTime=Time,Slide=True)
ReverseFactor=.2
T=MSMLib.EstimateTransitionMatrix(Counts+ReverseFactor*Counts.transpose())

#To save a sparse matrix, use:
scipy.io.mmwrite("./Data/tCount.mtx",Counts)
scipy.io.mmwrite("./Data/tProb.mtx",T)
#To load a sparse matrix, use:
#Counts=scipy.io.mmread("./Data/tCount.mtx")
#T=scipy.io.mmread("./Data/tProb.mtx")


#Step 7
#Get NumConfsPerState random conformations from each state, Save them as separate PDB files in the PDB directory.  Then save all the grabbed conformations as a .lh5 file (for fast loading and saving for analyses).
NumConfsPerState=10
RandomConfs=P1.SavePDBs(Assignments,"./PDB",NumConfsPerState,)

#To Load the trajectory later, use:
#RandomConfs=Trajectory.Trajectory.LoadFromLHDF("Data/%dConfsFromEachState.lh5"%NumConfsPerState)
CA=find(RandomConfs["Atoms"]=="CA")
rmsd=RandomConfs.CalcRMSD(C1,CA,CA).reshape((NumStates,NumConfsPerState)).mean(1)



#Step 8
EigAns=MSMLib.GetEigenvectors(T,NumEigen);
Populations=EigAns[1][:,0]

figure()
plot(rmsd,-log(Populations),'o')
title("Free Energy Versus RMSD [nm]")
ylabel("Free Energy")
xlabel("RMSD [nm]")




#How do I build a Macrostate MSM?
MAP=MSMLib.PCCA(T,50)
Assignments["Data"]=MAP[Assignments["Data"]]
#Now repeat any calculations using this new Macrostate Assignment data.
