
from msmbuilder.project import FahProjectBuilder

pb = FahProjectBuilder('./fah_style_data', '.xtc', './fah_style_data/native.pdb')
p = pb.convert()
print p
