#ifndef __QUALITY_H__
#define __QUALITY_H__
#include <stdint.h>

class Quality{
    public:
    	// default constructor
        Quality(int row, 
                int col, 
                int ldn, 
                int row_blk, 
                int col_blk, 
                float* input_mat,
                float* target_mat, 
                float* baseline_mat);

	    /* main APIs
	        Now each API call works on one tiling block only,
            and the block is indicated by i_blk_idx, j_blk_idx.
            block sizes are used by this->row_blk, this->col_blk.
         */
        float rmse(); 
        float error_rate();
        float error_percentage();
        float ssim();
        float pnsr();
        
        float rmse(int i_idx, int j_idx);
        float error_rate(int i_idx, int j_idx);
        float error_percentage(int i_idx, int j_idx);
        float ssim(int i_idx, int j_idx);
        float pnsr(int i_idx, int j_idx);
        
        void print_results(bool is_tiling, int verbose);

    private:
        struct DistStats{
            float mean;
            float sdev;
        };
        
        struct Unit{
            float rmse;
            float error_rate;
            float error_percentage;
            float ssim;
            float pnsr;
            
            // input array's statistic
            DistStats input_dist_stats;
        };
        void print_quality(Unit quality);

        // internal helper functions
        void get_minmax(int i_start, 
                        int j_start,
                        int row_size,
                        int col_size,
                        float* x, 
                        float& max,
                        float& min);
        float average(float* mat, int i_start, int j_start, int row_size, int col_size);
        float sdev(float* mat, int i_start, int j_start, int row_size, int col_size);
        float covariance(int i_start, int j_start, int row_size, int col_size);

        float rmse_kernel(int i_start, int j_start, int row_size, int col_size);
        float error_rate_kernel(int i_start, int j_start, int row_size, int col_size);
        float error_percentage_kernel(int i_start, int j_start, int row_size, int col_size);
        float ssim_kernel(int i_start, int j_start, int row_size, int col_size);
        float pnsr_kernel(int i_start, int j_start, int row_size, int col_size);
        int row;
	    int col;
	    int ldn;
        int row_blk;
        int col_blk;
        int row_cnt;
        int col_cnt;
        float* input_mat;
        float* target_mat;
	    float* baseline_mat;
};

#endif

