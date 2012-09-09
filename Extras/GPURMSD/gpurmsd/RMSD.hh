#ifndef __RMSD_CUH__
#define __RMSD_CUH__

//AUTHOR: Yutong Zhao | proteneer@gmail.com

//Please refer to Theobald et. al's paper on the RMSD algorithm here-in used:
//
//Liu P, Agrafiotis DK, & Theobald DL (2010) 
//"Fast determination of the optimal rotation matrix for macromolecular superpositions." 
//Journal of Computational Chemistry 31(7):1561-1563. [Open Access]
//
//Douglas L Theobald (2005) 
//"Rapid calculation of RMSDs using a quaternion-based characteristic polynomial." 
//Acta Crystallogr A 61(4):478-480. [Open Access, pdf]

class RMSD {

    public:
        // construct an RMSD object, 
        // h_X is stored in ADP format
        RMSD(int numAtoms, int numDimens, int numConfs, float* h_X);
        ~RMSD();

        // must always be set
        void set_rmsd_array(int numConfs, float *h_rmsds);

        // if set to non-NULL, automatically assumes we want to use only a subset of the atoms
        // when doing the RMSD calculations
        void set_subset_flag_array(int numConfs, int *h_subset_flag);
 
        //centers conformers and precompute_G();
        void center_and_precompute_G(); 

        void all_against_one_rmsd(int test_conf);
        void all_against_one_lprmsd(int test_conf);
            
        void print_params();

    private:
        
        // check GPU parameters
        void set_gpu_parameters();

        //kernel methods
        void center_conformers();
        void precompute_G();
        
        //system parameters
        const int numAtoms_;
        const int numConfs_;
        const int numBlocks_;
        const int size_;

        // sanity flags
        bool have_precomputed_G_;
        bool have_centered_;

        // GPU-specific variables
        static const int threadsperblock_ = 512;
        size_t capacity_;
        float compute_capability_;

        // pointers to data storage
        float *h_X_; // pointer host_data
        float *d_X_; // pointer to cuda data

        float *d_G_; // pointer to precomputed inner products

        float *h_rmsds_; // host rmsd storage
        //float *h_rot_mat_;
        int *h_subset_flag_; // host atom flags

        float *d_rmsds_; // pointer to device rmsds (all against 1)
        int *d_subset_flag_; // host subset flags
};

#endif
