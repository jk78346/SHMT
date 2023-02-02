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
        
        //float max(float* mat, int i_start, int j_start, int row_size, int col_size);
        float in_max(){ return this->max(this->input_mat, 
                                         0,
                                         0,
                                         this->row,
                                         this->col); };
        float in_min(){ return this->min(this->input_mat,
                                         0,
                                         0,
                                         this->row,
                                         this->col); };
        float in_mean(){ return this->average(this->input_mat,
                                              0,
                                              0,
                                              this->row,
                                              this->col); };
        float in_sdev(){ return this->sdev(this->input_mat,
                                           0,
                                           0,
                                           this->row,
                                           this->col); };
        float in_entropy(){ return this->entropy(this->input_mat,
                                                 0,
                                                 0,
                                                 this->row,
                                                 this->col); };
        
        float in_max(int i_idx, int j_idx){ return this->max(this->input_mat, 
                                                             i_idx*this->row_blk,
                                                             j_idx*this->col_blk,
                                                             this->row_blk,
                                                             this->col_blk); };
        float in_min(int i_idx, int j_idx){ return this->min(this->input_mat, 
                                                             i_idx*this->row_blk,
                                                             j_idx*this->col_blk,
                                                             this->row_blk,
                                                             this->col_blk); };
        float in_mean(int i_idx, int j_idx){ return this->average(this->input_mat, 
                                                                  i_idx*this->row_blk,
                                                                  j_idx*this->col_blk,
                                                                  this->row_blk,
                                                                  this->col_blk); };
        float in_sdev(int i_idx, int j_idx){ return this->sdev(this->input_mat, 
                                                               i_idx*this->row_blk,
                                                               j_idx*this->col_blk,
                                                               this->row_blk,
                                                               this->col_blk); };
        float in_entropy(int i_idx, int j_idx){ return this->entropy(this->input_mat, 
                                                                     i_idx*this->row_blk,
                                                                     j_idx*this->col_blk,
                                                                     this->row_blk,
                                                                     this->col_blk); };
        void print_results(bool is_tiling, int verbose);
        void print_histogram(float* input);

        int get_row(){ return this->row; }
        int get_col(){ return this->col; }
        int get_row_blk(){ return this->row_blk; }
        int get_col_blk(){ return this->col_blk; }
        int get_row_cnt(){ return this->row_cnt; }
        int get_col_cnt(){ return this->col_cnt; }

    private:
        struct DistStats{
            float max;
            float min;
            float mean;
            float sdev;
            float entropy;
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

        float max(float* mat, int i_start, int j_start, int row_size, int col_size);
        float min(float* mat, int i_start, int j_start, int row_size, int col_size);
        float average(float* mat, int i_start, int j_start, int row_size, int col_size);
        float sdev(float* mat, int i_start, int j_start, int row_size, int col_size);
        float covariance(float* x, float* y, int i_start, int j_start, int row_size, int col_size);
        float entropy(float* mat, int i_start, int j_start, int row_size, int col_size);

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

