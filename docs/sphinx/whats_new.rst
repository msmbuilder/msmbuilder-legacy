What's New
==========

2.8 Changelog
-------------
   - Ported code to use MDTraj. This eliminates the need for MSMBuilder to
     directly handle the loading and saving of trajectories, and gives us
     much more cross-format support.
   - New sphinx documentation at http://msmbuilder.org
   - MSMBuilder is now Python 3 compatible.
   - MSMBuilder now supports windows.
   
2.6 Changelog
-------------

   - Migrated from SVN to Github for version control. The source code can 
     now be found at https://github.com/SimTk/msmbuilder

   - Renamed all the functions in MSMLib to be pep8 compliant. 
     Aliases have been added for the old names.  Note: Trajectory.py is 
     still not pep8 compliant, but will be fixed by release 3.0.

   - Split MSMLib into 2 files: MSMLib and msm_analysis. msm_analysis contains 
     all the code for analyzing MSMs (eigenvector calculation, sampling, etc). 
     MSMLib contains code for building models, while msm_analysis contains 
     code for working with models.

   - Removed AssignMPI.py

   - Added more unit tests and Travis automated unit testing.

   - Updated most of MSMBuilder to be PEP8 compliant.

   - Rewrote code for estimating reversible maximum likelihood count matrix.  
     The new code should be faster and easier to read.

   - New code implementing min-cut/max-flow cut-based reaction coordinate 
     learning from MSMs (cfep.py). Beta version.

   - New code for the computation of MSM hub scores.

   - Switched to YAML for storing project information.  

   - Added script RebuildProject.py to construct YAML ProjectInfo file.

   - Deleted the :class:`Serializer` class.

   - New msmbuilder.io module for storing output.  

   - Added support for the BACE coarse graining algorithm.

   - Better error checking for assignments arrays.  
   
   - Updates to tutorials.   

   - Rewrote lumping code for improved readability.  

2.5.1 Changelog
---------------

   - Updates to tutorials.

   - Moved Ward clustering and SCRE to AdvancedMethods tutorial.
   

2.5 Changelog
-------------

   - The libraries are distance metric agnostic, you can use your own 
     new distance metric without having to change the clustering/assignment code.
     
   - Dihedral, Contact (residues and/or atoms), hybrid distance metric code

   - New clustering algorithms (CLARANS / subsampled CLARANS)

   - Flux-based PCCA+.

   - Support for Hierarchical Clustering (e.g. Ward).

   - SCRE rate matrix estimation.

2.0.4 Changelog
---------------

   - Improvements in PCCA, PCCA+ lead to better macrostate definition.

2.0.3 Changelog
---------------
   - Bug fix: In ConvertDataToHDF, the Stride option was not being used 
     for non-FAH style datasets. 
   
   - Fixed unicode strings were causing scipy issues on certain platforms.

2.0.2 Changelog
---------------


   - Fixed issue with AtomIndices in UpdateProjectToHDF

   - Using dtype='int' for Assignments, consistent throughout MSMBuilder.

   - CalculateImpliedTimescales now allows users to exit through control+C

   - Fixed some tabbing issues in MSMLib.py

2.0.1 Changelog
---------------

   - Fixed a bug in ConvertProjectToHDF that prevented MSMBuilder from seeing 
     multiple XTC files in a single directory. 

   - Fixed carriage return issues in UpdateProjectToHDF 

   - Fixed minor bugs in Project class. 

   - If Project cannot find its PDB, it also will look in the current directory. 
     This helps when the absolute path of a Project changes. 

   - Unit tests. 

   - Removed unnecessary files, reducing the package size to 10MB. 

