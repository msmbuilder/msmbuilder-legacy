Frequently Asked Questions
==========================

1. How do I decrease / increase the number of threads used during
clustering, assignment, and rmsd calculation?

    Set the

    ::

        OMP_NUM_THREADS
    
    environment variable to the desired number of threads. In linux, you
    would type (or add to your bashrc):

    ::

        export OMP_NUM_THREADS=6 

2. I see the following error. What do I do?

    ::

        #004: H5Z.c line 1095 in H5Z_pipeline(): required filter is not registered

    You are trying to read an HDF5 file that was written using a
    different PyTables installation. Your current PyTables installation is
    likely missing the compression algorithm (filter) required to read the
    file. The solution is to find a version of Pytables that has the old
    compression algorithm (filter) and use MSMBuilder to read and then
    re-write the trajectories (by default, MSMBuilder uses the PyTables
    BLOSC compression). To do this (for a single File), you would so
    something like:

    ::

        from msmbuilder import Trajectory
        R1 = Trajectory.LoadFromLHDF(Filename)
        R1.SaveToLHDF(NewFilename)

3. I received an “Illegal instruction” error. What does this mean?

    MSMBuilder2 requires an SSE3 compatible processor when Clustering
    and calculating RMSDs. Any processor built after 2006 should have the
    necessary instructions.

4. In my implied timescales plot, I see unphysically slow timescales.

    The current estimators for transition matrices are somewhat
    sensitive to poor statistics. The hybrid k-centers / k-kmedoids
    clustering focuses on providing the best possible clustering–without
    regard to the quality of the resulting statistics. Thus, to get more
    precise timescales, you may have to find a way to achieve better
    statistics. Here are a few ideas:

    #. Collect longer trajectories.

    #. Use fewer states. Also, by increasing the number of local and global
       k-medoid updates, you can often increase the accuracy of your
       clustering while simultaneously lowering the number of states.

    #. Subsample your data when clustering.

    #. Skip the initial k-centers step of clustering, instead using randomly
       selected conformations. This generally leads to poorer clustering
       quality, but considerably better statistics in each state. (Thus, the
       clusters will be much more localized to regions of high population
       density.) This can be achieved by setting “-r 0” when clustering.

    #. Use Ward clustering

5. Why are there -1s in my Assignments matrix?

    We use -1 as a “padding” element in Assignment matrices. Suppose
    your project has maximum trajectory length of 100. If trajectory 0 has
    length 50, then A[0,50:] should be a vector of -1. Furthermore, when you
    perform trimming to ensure (strong) ergodicity, futher -1s could be
    introduced at the start or finish of the trajectory. Finally, if Ergodic
    trimming was performed with count matrices estimated using a sliding
    window, you could even see something like: -1 -1 -1 x -1 -1 y z … This
    is because sliding window essentially splits your trajectory into
    independent subtrajectories–one for each possible window starting
    position. “x” then marks the start of one of these subtrajectories.

6. When building MSMBuilder, I see an LPRMSD error. What should I do?

    ::

        ***************************************************************************
        WARNING: The C extension 'LPRMSD' could not be compiled.
        This may be due to a failure to find the BLAS libraries
        ***************************************************************************

    Don’t worry. This module is not used by any of the standard
    MSMBuilder features.

7. What is the difference between PCCA+ and FPCCA+?

    FPCCA+ is PCCA+ with a different choice of eigenvectors to model. In
    particular, FPCCA+ uses a criterion based on both timescale *and*
    eigenvector flux.

8. Should I use FPCCA+ or PCCA+?

    First, note that FPCCA+ is more “lossy” or “coarse-grained” than
    PCCA+. By discarding slow but high-flux eigenvectors, you are losing
    some information from your microstate model. Essentially, the choice
    between FPCCA+ and PCCA+ depends on how much you weight model accuracy
    versus model simplicity.

Q9. I see warnings when using PCCA+:

    ::

        ComplexWarning: Casting complex values to real discards the imaginary part
        RuntimeWarning: invalid value encountered in cdouble_scalars
        Warning: constraint violation detected.
        f = nan

    This is probably due to PCCA+ finding a “degenerate” state
    decomposition, where one of your macrostates is empty. Usually, the
    minimization procedure should eventually find a feasible point with the
    correct number of states. Be sure to check that your resulting state
    decomposition makes sense.

10. How do I make an MSM movie?

    To build a movie, you just Sample states from the model
    (MSMLib.Sample). Then you sample conformations from each state (Project
    . GetRandomConfsFromState). Then you append each frame to a PDB file
    (Conformation.SavePDB or Trajectory.SavePDB). After you have the PDBs,
    you can use either VMD or pymol for movie making.

11. On an OSX Lion Machine, clustering fails with an “Abort Trap” error
message. What do I do?

    This is due to a known bug in OSX Lion’s support for OpenMP (see
    https://discussions.apple.com/thread/3786045?start=0&tstart=0). As a
    workaround, you can simply

    ::

        export OMP_NUM_THREADS=1

    to disable OpenMP support during clustering. This should eliminate the
    problem, but it limits you to single core clustering.