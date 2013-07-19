
def GetAtomIndicesAndPosition(C1):
    a=C1["AtomNames"]
    aC=np.where(a=="C")[0]
    aN=np.where(a=="N")[0]
    aCA=np.where(a=="CA")[0]
    X=C1["XYZ"]
    NumResi=C1.GetNumberOfResidues()
    return(NumResi,a,aC,aN,aCA,X)


def GetAllPsi(C1,NumResi=None,a=None,aC=None,aN=None,aCA=None,X=None):
    if a==None:
        NumResi,a,aC,aN,aCA,X=GetAtomIndicesAndPosition(C1)
    Data=[]
    for i in xrange(NumResi-1):
        print(i)
        x0=X[aN[i]]
        x1=X[aCA[i+1]]
        x2=X[aC[i+1]]
        x3=X[aN[i+1]]
        Psi=Torsion(x0,x1,x2,x3)
        Data.append(Psi)
    return(np.array(Data))


def GetAllPhi(C1,NumResi=None,a=None,aC=None,aN=None,aCA=None,X=None):
    if a==None:
        NumResi,a,aC,aN,aCA,X=GetAtomIndicesAndPosition(C1)
    Data=[]
    for i in xrange(NumResi-1):
        print(i)
        x0=X[aC[i]]
        x1=X[aN[i+1]]
        x2=X[aCA[i+1]]
        x3=X[aC[i+1]]
        Phi=Torsion(x0,x1,x2,x3)
        Data.append(Phi)
    return(np.array(Data))

def GetAllPsiDipeptide(C1,NumResi=None,a=None,aC=None,aN=None,aCA=None,X=None):
    if a==None:
        NumResi,a,aC,aN,aCA,X=GetPsiAtomIndicesAndPosition(C1)
    
    a0=aN[0]
    a1=aCA[0]#Corrected because resi 0 has no CA
    a2=aC[1]
    a3=aN[1]  
    x0=X[a0]
    x1=X[a1]
    x2=X[a2]
    x3=X[a3]
    Psi=Torsion(x0,x1,x2,x3)
    return(np.array([Psi]))

def GetAllPhiDipeptide(C1,NumResi=None,a=None,aC=None,aN=None,aCA=None,X=None):
    if a==None:
                NumResi,a,aC,aN,aCA,X=GetPhiAtomIndicesAndPosition(C1)
    a0=aC[0]
    a1=aN[0]
    a2=aCA[0]#Corrected because resi 0 has no CA
    a3=aC[1]
    x0=X[a0]
    x1=X[a1]
    x2=X[a2]
    x3=X[a3]
    Phi=Torsion(x0,x1,x2,x3)
    return(np.array([Phi]))

def Torsion(x0,x1,x2,x3,Degrees=True):
    """Calculate the signed dihedral angle between 4 positions."""
    #Calculate Bond Vectors b1, b2, b3
    b1=x1-x0
    b2=x2-x1
    b3=x3-x2

    #Calculate Normal Vectors c1,c2.  This numbering scheme is idiotic, so care.
    c1=np.cross(b2,b3)
    c2=np.cross(b1,b2)

    Arg1=np.dot(b1,c1)
    Arg1*=np.linalg.norm(b2)

    Arg2=np.dot(c2,c1)

    phi=np.arctan2(Arg1,Arg2)

    if Degrees==True:
        phi*=180./np.pi
    return(phi)

def get_angles(pdb_path = "./native.pdb", project_path = "./ProjectInfo.yaml", assignments_path = "./Macro4/MacroAssignments.h5",
                samplesPerState = 10):
    # Load conformation and project
    C1=Conformation.load_from_pdb(pdb_path)
    P1=Project.load_from(project_path)

    # Extract information from the topology file
    NumResi,a,aC,aN,aCA,X=GetAtomIndicesAndPosition(C1)


    def GetPhi(X):
        return GetAllPhiDipeptide(C1,NumResi=NumResi,a=a,aC=aC,aN=aN,aCA=aCA,X=X)
    def GetPsi(X):
        return GetAllPsiDipeptide(C1,NumResi=NumResi,a=a,aC=aC,aN=aN,aCA=aCA,X=X)

    # Load assignments
    assign = io.loadh(assignments_path, "arr_0")
    numStates = assign.max() - assign.min() + 1

    # Lambda function to get max number of conformations per macrostate
    max_per_state = lambda x: len(np.where(assign == x)[0])
    # Number of conformations to sample
    if samplesPerState < 0:
        num_confs = [max_per_state(i) for i in xrange(numStates)]
    else:
        num_confs = [min(samplesPerState, max_per_state(i)) for i in xrange(numStates)]
    # Total number of conformations
    num_confs_tot = sum(num_confs)

    print(num_confs)

    # Sample conformations
    confs = P1.get_random_confs_from_states(assign, range(numStates), num_confs, replacement = False)

    # Initialize arrays
    phi = np.zeros(num_confs_tot)
    psi = np.zeros(num_confs_tot)
    ind = np.array(num_confs, dtype=np.int)

    # Loop through conformations and get phi and psi angles
    i_tot = 0
    for c in confs:
        cxyz = c["XYZList"]
        for i_conf in xrange(len(cxyz)):
            phi[i_tot] = GetPhi(c["XYZList"][i_conf])
            psi[i_tot] = GetPsi(c["XYZList"][i_conf])
            i_tot+=1

    io.saveh("Dihedrals.h5", Phi = phi, Psi = psi, StateIndex = ind)

    return phi, psi


import numpy as np
from msmbuilder import Conformation,Project,io
from argparse import ArgumentParser
import os


def main():
    parser = ArgumentParser(os.path.split(__file__)[1], description = '''
    A simple script to extract Phi and Psi angles for Alanine dipeptide.
    This script will only work with Alanine dipeptide due to its unique naming convention.''')

    parser.add_argument('--pdb', dest = 'pdb_path',
                    help = 'The path to the topology file. Default: native.pdb',
                    default = 'native.pdb', metavar = 'pdb_path')
    parser.add_argument('-p', '--project', dest='project_path',
                    help = 'MSMBuilder project file. Default: ProjectInfo.yaml',
                    default = 'ProjectInfo.yaml', metavar = 'project')
    parser.add_argument('-a', dest='assignments_path',
                    help = 'Path to Macrostate assignments. Default: Macro4/MacroAssignments.h5',
                    default = 'Macro4/MacroAssignments.h5', metavar = 'assignments')
    parser.add_argument('-n', dest = 'samples',
                    help = 'The number of states to sample per macrostate. Use -1 for all. Default: 1000',
                    default=1000, metavar = 'samples', type=int)

    args = parser.parse_args()
    get_angles(pdb_path = args.pdb_path, project_path = args.project_path,
                    assignments_path = args.assignments_path, samplesPerState = args.samples)

if __name__ == '__main__':
    main()

