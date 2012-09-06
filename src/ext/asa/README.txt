################# ALGORITHM DESCRIPTION ###########################
#
# The C-extension module asa implements the Shrake and Rupley
# algorithm. Shrake, A; Rupley, JA. (1973) J Mol Biol 79 (2): # with the Golden Section Spiral algorithm to generate the sphere
# points
#
# The basic idea is to great a mesh of points representing the surface
# of each atom (at a distance of the van der waals radius plus the probe
# radis from the nuclei), and then count the number of such mesh points
# that are on the molecular surface -- i.e. not within the radius of another
# atom. Assuming that the points are evenly distributed, the number of points
# is directly proportional to the accessible surface area (its just 4*pi*r^2
# time the fraction of the points that are accessible).

# There are a number of different ways to generate the points on the sphere --
# possibly the best way would be to do a little "molecular dyanmis" : puts the
# points on the sphere, and then run MD where all the points repel one another
# and wait for them to get to an energy minimum. But that sounds expensive.

# This code uses the golden section spiral algorithm
# (picture at http://xsisupport.com/2012/02/25/evenly-distributing-points-on-a-sphere-with-the-golden-sectionspiral/)
# where you make this spiral that traces out the unit sphere and then put points
# down equidistant along the spiral. It's cheap, but not perfect

# the Gromacs utility g_sas uses a slightly different algorithm for generating
# points on the sphere, which is based on an icosahedral tesselation.
# roughly, the icosahedral tesselation works something like this
# http://www.ziyan.info/2008/11/sphere-tessellation-using-icosahedron.html

###################################################################
