import __main__
import sys, time, os
import pymol
pymol.finish_launching()

if not os.path.exists(OutDir):
    os.mkdir(OutDir)

GetFilename=lambda i,j: PDBDir+"State%d-%d.pdb"%(i,j)
GetName=lambda i,j: "State%d-%d"%(i,j)
time.sleep(.1) 

pymol.cmd.load(NativeFilename, "native")
pymol.cmd.util.chainbow("native")

for i in range(NumStates):
    for j in range(NumConfs):
        print(GetName(i,j))
        pymol.cmd.load(GetFilename(i,j), GetName(i,j))
        pymol.cmd.util.chainbow(GetName(i,j))
        pymol.cmd.fit(GetName(i,j),"native")

pymol.cmd.zoom()
pymol.cmd.hide("lines")
pymol.cmd.show("cartoon")

#disable all objects from viewing
pymol.cmd.disable("all")

for i in range(NumStates):
    for j in range(NumConfs):
            #enable all of state i
            pymol.cmd.enable(GetName(i,j))
    time.sleep(.1)
    print("ray")
    pymol.cmd.ray(600,600)
    print("done")
    time.sleep(.1)
    pymol.cmd.png(OutDir+"State%d.png"%i)
    time.sleep(.1)
    for j in range(NumConfs):
        pymol.cmd.disable(GetName(i,j))

pymol.cmd.quit()
