import Trajectory
import Serializer

def Convert(InFilename,OutFilename):
    """A simple conversion utility that fixes the trajectory key names after msmbuilder update."""
    R1=Serializer.Serializer.LoadFromHDF(InFilename)
    R1["ChainID"]=R1.pop("Chains")
    R1["AtomNames"]=R1.pop("Atoms")
    R1["AtomID"]=R1.pop("AtomNums")
    R1["ResidueID"]=R1.pop("ResNums")
    R1["ResidueNames"]=R1.pop("ResNames")
    R1.pop("Title")
    R1.pop("nResi")
    R1.pop("nAtoms")
    R1.SaveToHDF(OutFilename)
