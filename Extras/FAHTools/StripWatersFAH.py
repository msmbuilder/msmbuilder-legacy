import os
import string

def run():
    ProjNum=10424
    NumClones=1
    NumRuns=50000

    ServerIP="171.64.65.79"
    IntServerIP=int(string.replace(ServerIP,".",""))
    ProjName="PROJ%d"%ProjNum
    CleanDir="./CleanData/"
    OutDir="%s/%s"%(CleanDir,ProjName)
    SourceDir="./data/SVR%d/%s/"%(IntServerIP,ProjName)
    NDXFile="/home/server.171.64.65.79/server2/kyleb/10424/notsol.ndx"
    TPRFile=SourceDir+"/RUN0/CLONE0/frame0.tpr"

    MinGens=77
    MaxGens=1000

    CreateFolders(CleanDir,OutDir,NumRuns,NumClones)
    StripWaters(OutDir,NumRuns,NumClones,SourceDir,NDXFile,TPRFile,MinGens,MaxGens)

def CreateFolders(CleanDir,OutDir,NumRuns,NumClones):
    """Create folders to store preprocessed XTC files from a FAH project."""    

    try:
        os.mkdir(CleanDir)
        os.mkdir(OutDir)
    except OSError:
        print("Directory %s already found.  No directories will be created."%OutDir)
        return

    for i in range(NumRuns):
        DirName="./%s/RUN%d/"%(OutDir,i)
        os.mkdir(DirName)
        for j in range(NumClones):
            DirName="./%s/RUN%d/CLONE%d/"%(OutDir,i,j)
            print("Making Directory %s"%DirName)
            os.mkdir(DirName)

def StripWaters(OutDir,NumRuns,NumClones,SourceDir,NDXFile,TPRFile,MinGens=1,MaxGens=1000):
    """Preprocess XTC files from a FAH project.  """

    for i in range(NumRuns):
        print("RUN%d"%i)
        for j in range(NumClones):
            DirName="%s/RUN%d/CLONE%d/"%(OutDir,i,j)
            InDirName="%s/RUN%d/CLONE%d/"%(SourceDir,i,j)
            CurrentNumGens=0
            for k in range(MaxGens):
                InXTCFile=InDirName+"/frame%d.xtc"%k
                OutXTCFile=DirName+"/frame%d.xtc"%k
                if os.path.exists(InXTCFile):
                    CurrentNumGens+=1
            if CurrentNumGens<=MinGens:
                CurrentNumGens=0
            for k in range(CurrentNumGens):
                InXTCFile=InDirName+"/frame%d.xtc"%k
                OutXTCFile=DirName+"/frame%d.xtc"%k
                if os.path.exists(OutXTCFile):
                    print("Skipping %s; already copied."%OutXTCFile)
                else:
                    print("Copying %s."%OutXTCFile)
                    cmd="trjconv -f %s -n %s -pbc whole -o %s -s %s"%(InXTCFile,NDXFile,OutXTCFile,TPRFile)
                    print(cmd)
                    os.system(cmd)


if __name__ == "__main__":
    print """
    Strip Waters and Preprocess XTCs in FAH project.
    """
    run()
