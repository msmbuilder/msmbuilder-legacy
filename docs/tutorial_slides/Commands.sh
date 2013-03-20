cd $PREFIX/share/msmbuilder_tutorial 
tar -xvf XTC.tar

ConvertDataToHDF.py  -s native.pdb -i XTC

Cluster.py rmsd hybrid -d 0.045 -l 50

CalculateImpliedTimescales.py -l 1,25 -i 1 -o Data/ImpliedTimescales.dat -e 5

PlotImpliedTimescales.py -d 1. -i Data/ImpliedTimescales.dat

BuildMSM.py -l 1

PCCA.py -n 4 -a Data/Assignments.Fixed.h5 -t Data/tProb.mtx -o Macro4/ -A PCCA+

python PlotDihedrals.py Macro4/MacroAssignments.h5

CalculateImpliedTimescales.py -l 1,25 -i 1 \
-o Macro4/ImpliedTimescales.dat -a Macro4/MacroAssignments.h5 -e 3

PlotImpliedTimescales.py -i Macro4/ImpliedTimescales.dat -d 1

BuildMSM.py -l 6 -a Macro4/MacroAssignments.h5 -o Macro4/   

SaveStructures.py -s -1 -a Macro4/MacroAssignments.h5  -c 1 -S sep
pymol PDBs/State0-0.pdb PDBs/State1-0.pdb PDBs/State2-0.pdb PDBs/State3-0.pdb
