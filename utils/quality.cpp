#include "utils.h"
#include "quality.h"
#include "math.h"
#include <map>
#include <vector>
#include <float.h>
#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

Quality::Quality(int m, 
                 int n, 
                 int ldn, 
                 int row_blk, 
                 int col_blk,
                 float* input_mat,
                 float* x,
                 float* y){
	this->row          = m;
	this->col          = n;
	this->ldn          = ldn;
    this->row_blk      = row_blk;
    this->col_blk      = col_blk;
    assert(row % row_blk == 0);
    assert(col % col_blk == 0);
    this->row_cnt = row / row_blk;
    this->col_cnt = col / col_blk;
    assert(this->row_cnt >= 1);
    assert(this->col_cnt >= 1);
    this->input_mat = input_mat;
    this->target_mat   = x;
	this->baseline_mat = y;
    
    this->result_pars.resize(this->row_cnt * this->col_cnt);

    std::cout << __func__ << ": calculating metrices..." << std::endl;
    // global quality
    std::cout << "global quality..." << std::endl;
    this->common_kernel(this->result, 0, 0, this->row, this->col);
    this->common_stats_kernel(this->result.input_dist_stats, 
                              this->input_mat, 
                              0, 
                              0, 
                              this->row, 
                              this->col);
    bool is_tiling = (this->row > this->row_blk)?true:false;
    
    if(is_tiling){
        // tiling quality
        for(int i = 0 ; i < this->row_cnt ; i++){
            for(int j = 0 ; j < this->col_cnt ; j++){
                int idx = i * this->col_cnt + j;
                std::cout << "tiling quality(" << i << ", " << j << ")..." << std::endl;
                this->common_kernel(this->result_pars[idx], 
                                    i*this->row_blk,
                                    j*this->col_blk,
                                    this->row_blk,
                                    this->col_blk);
              this->common_stats_kernel(this->result_pars[idx].input_dist_stats, 
                                        this->input_mat, 
                                        i*this->row_blk, 
                                        j*this->col_blk, 
                                        this->row_blk, 
                                        this->col_blk);
            }
        }    
    }
}

void Quality::common_stats_kernel(DistStats& stats, float* x, int i_start, int j_start, int row_size, int col_size){
    float max = FLT_MIN;
    float min = FLT_MAX;
    double sum = 0.0;
	double square_sum = 0.0;
    float entropy = 0.0;
    std::map<float, long int>counts;
    std::map<float, long int>::iterator it;
    it = counts.begin();
    int elements = row_size * col_size;
   
    // max, min, average, entropy(1)
//#pragma omp parallel for
    for(int i = i_start ; i < i_start+row_size ; i++){
        for(int j = j_start ; j < j_start+col_size ; j++){
            int idx = i*this->ldn+j;
            float tmp = x[idx];
            sum += tmp;
            max = (tmp > max)?tmp:max;
            min = (tmp < min)?tmp:min;
            counts[tmp]++;
        }
    }
    stats.max = max;
    stats.min = min;
	stats.mean = (float)(sum / (double)(elements));
    
    // sdev
#pragma omp parallel for
    for(int i = i_start ; i < i_start+row_size ; i++){
        for(int j = j_start ; j < j_start+col_size ; j++){
			square_sum += pow(x[i*this->ldn+j] - stats.mean, 2);
        }
    }
	stats.sdev = pow((float)(sum / (double)(elements)), 0.5);

    // entropy(2)
    while(it != counts.end()){
        float p_x = (float)it->second/elements;
        if(p_x > 0){
            entropy-= p_x*log(p_x)/log(2);
        }
        it++;
    }
    stats.entropy = entropy;
}

void Quality::common_kernel(Unit& result, int i_start, int j_start, int row_size, int col_size){
    
    double mse = 0;
	double mean; 
    float baseline_max = FLT_MIN, baseline_min = FLT_MAX;
    float target_max = FLT_MIN, target_min = FLT_MAX;
    double rate = 0;
	int cnt = 0;
	int error_percentage_cnt = 0;
    double baseline_sum = 0.0;
#pragma omp parallel for
    for(int i = i_start ; i < i_start+row_size ; i++){
		for(int j = j_start ; j < j_start+col_size ; j++){
			int idx = i*this->ldn+j;
			baseline_sum += this->baseline_mat[idx];
            baseline_max = 
                (this->baseline_mat[idx] > baseline_max)?
                this->baseline_mat[idx]:
                baseline_max;
            baseline_min = 
                (this->baseline_mat[idx] < baseline_min)?
                this->baseline_mat[idx]:
                baseline_min;
            target_max = 
                (this->target_mat[idx] > target_max)?
                this->target_mat[idx]:
                target_max;
            target_min = 
                (this->target_mat[idx] < target_min)?
                this->target_mat[idx]:
                target_min;
            mse = 
                (mse * cnt + 
                 pow(this->target_mat[idx] - this->baseline_mat[idx], 2)) 
                / (cnt + 1);
			rate = 
                (rate * cnt + 
                 fabs(this->target_mat[idx] - this->baseline_mat[idx])) 
                / (cnt + 1);
			if(fabs(this->target_mat[idx] - this->baseline_mat[idx]) > 1e-8){
				error_percentage_cnt++;
			}
			cnt++;
		}
	}

    mean = (float)(baseline_sum / (double)(row_size*col_size));
    
    assert(error_percentage_cnt <= (row_size * col_size));
	
    // SSIM default parameters
	int    L = 255.0; // 2^(# of bits) - 1
	float k1 = 0.01;
	float k2 = 0.03;
	float c1 = 6.5025;  // (k1*L)^2
	float c2 = 58.5225; // (k2*L)^2
    if(target_max > 255.){
        std::cout << __func__ 
                  << ": [WARN] should ignore ssim since array.max = " 
                  << target_max << std::endl;
    }

	// update dynamic range
	L = fabs(target_max - target_min); 
	c1 = (k1*L)*(k1*L);
	c2 = (k2*L)*(k2*L);

	// main calculation
	float ssim = 0.0;
	
	float ux = this->average(this->target_mat, i_start, j_start, row_size, col_size);
	float uy = this->average(this->baseline_mat, i_start, j_start, row_size, col_size);
	float vx = this->sdev(this->target_mat, i_start, j_start, row_size, col_size);
	float vy = this->sdev(this->baseline_mat, i_start, j_start, row_size, col_size);
	float cov = this->covariance(this->target_mat, this->baseline_mat, i_start, j_start, row_size, col_size);

	ssim = ((2*ux*uy+c1) * (2*cov+c2)) / ((pow(ux, 2) + pow(uy, 2) + c1) * (pow(vx, 2) + pow(vy, 2) + c2));
    //assert(ssim >= 0.0 && ssim <= 1.);
    if(ssim < 0.0 || ssim > 1.){
        std::cout << __func__ << " [WARN] ssim is out of bound. It may due to wrong value or"
                  << " the result of this benchmark isn't suitable for doing ssim (float type)"
                  << std::endl;
    }
	
    // assign results
    result.rmse = sqrt(mse);
	result.rmse_percentage = (result.rmse/mean) * 100.0;
	result.error_rate = (rate / mean) * 100.0;
    result.error_percentage = ((float)error_percentage_cnt / (float)(row_size*col_size)) * 100.0;
    result.ssim = ssim;
	result.pnsr = 20*log10(baseline_max) - 10*log10(mse/mean);
}

void Quality::get_minmax(int i_start, 
                         int j_start,
                         int row_size,
                         int col_size, 
                         float* x,
                         float& max, 
                         float& min){
	float curr_max = FLT_MIN;
	float curr_min = FLT_MAX;
	for(int i = i_start ; i < i_start+row_size ; i++){
		for(int j = j_start ; j < j_start+col_size ; j++){
			if(x[i*this->ldn+j] > curr_max){
				curr_max = x[i*this->ldn+j];
			}
			if(x[i*this->ldn+j] < curr_min){
				curr_min = x[i*this->ldn+j];
			}
		}
	}
	max = curr_max;
	min = curr_min;
}

float Quality::max(float* x,
                   int i_start,
                   int j_start,
                   int row_size,
                   int col_size){
    float ret = FLT_MIN;
    for(int i = i_start ; i < i_start+row_size ; i++){
        for(int j = j_start ; j < j_start+col_size ; j++){
            if(x[i*this->ldn+j] > ret){
                ret = x[i*this->ldn+j];
            }
        }
    }
    return ret;
}

float Quality::min(float* x,
                   int i_start,
                   int j_start,
                   int row_size,
                   int col_size){
    float ret = FLT_MAX;
    for(int i = i_start ; i < i_start+row_size ; i++){
        for(int j = j_start ; j < j_start+col_size ; j++){
            if(x[i*this->ldn+j] < ret){
                ret = x[i*this->ldn+j];
            }
        }
    }
    return ret;
}

float Quality::average(float* x, 
                       int i_start,
                       int j_start, 
                       int row_size,
                       int col_size){
	double sum = 0.0;
	for(int i = i_start ; i < i_start+row_size ; i++){
		for(int j = j_start ; j < j_start+col_size ; j++){
			sum += x[i*this->ldn+j];
		}
	}
	return (float)(sum / (double)(row_size*col_size));
}

float Quality::sdev(float* x, int i_start, int j_start, int row_size, int col_size){
	double sum = 0;
	float ux = this->average(x, i_start, j_start, row_size, col_size);
	for(int i = i_start ; i < i_start+row_size ; i++){
		for(int j = j_start ; j < j_start+col_size ; j++){
			sum += pow(x[i*this->ldn+j] - ux, 2);
		}
	}
	return pow((float)(sum / (double)(row_size*col_size)), 0.5);
}

float Quality::entropy(float* x, int i_start, int j_start, int row_size, int col_size){
    float ret = 0.0;
    std::map<float, long int>counts;
    std::map<float, long int>::iterator it;
    for(int i = i_start ; i < i_start+row_size ; i++){
        for(int j = j_start;  j < j_start+col_size ; j++){
            counts[x[i*this->ldn+j]]++;
        }    
    }
    it = counts.begin();
    int elements = row_size * col_size;
    while(it != counts.end()){
        float p_x = (float)it->second/elements;
        if(p_x > 0){
            ret-= p_x*log(p_x)/log(2);
        }
        it++;
    }
    return ret;
}

float Quality::covariance(float* x, float* y, int i_start, int j_start, int row_size, int col_size){
	double sum = 0;
	float ux = this->average(this->target_mat, i_start, j_start, row_size, col_size);
	float uy = this->average(this->baseline_mat, i_start, j_start, row_size, col_size);
	for(int i = i_start ; i < i_start+row_size ; i++){
		for(int j = j_start ; j < j_start+col_size ; j++){
			sum += (x[i*this->ldn+j] - ux) * (y[i*this->ldn+j] - uy);
		}
	}
	return (float)(sum / (double)(row_size*col_size));
}

float Quality::rmse(){
    return this->result.rmse;
}

float Quality::rmse(int i, int j){
    return this->result_pars[i * this->col_cnt + j].rmse;
} 

float Quality::rmse_percentage(){
    return this->result.rmse_percentage;
}

float Quality::rmse_percentage(int i, int j){
    return this->result_pars[i * this->col_cnt + j].rmse_percentage;
} 

float Quality::error_rate(){
    return this->result.error_rate;
}

float Quality::error_rate(int i, int j){
    return this->result_pars[i * this->col_cnt + j].error_rate;
}

float Quality::error_percentage(){
    return this->result.error_percentage;
}

float Quality::error_percentage(int i, int j){
    return this->result_pars[i*this->col_cnt + j].error_percentage;
}

float Quality::ssim(){
    return this->result.ssim;
}

float Quality::ssim(int i, int j){
    return this->result_pars[i*this->col_cnt + j].ssim;
}

float Quality::pnsr(){
    return this->result.pnsr;
}

float Quality::pnsr(int i, int j){
    return this->result_pars[i*this->col_cnt + j].pnsr;
}

void Quality::print_quality(Unit quality){
    std::cout << "(" << quality.input_dist_stats.max << ", ";
    std::cout << quality.input_dist_stats.min << ", ";
    std::cout << quality.input_dist_stats.mean << ", ";
    std::cout << quality.input_dist_stats.sdev << ", ";
    std::cout << quality.input_dist_stats.entropy << ") | ";
    std::cout << quality.rmse << "\t, ";
    std::cout << quality.rmse_percentage << "\t, ";
    std::cout << quality.error_rate << "\t, ";
    std::cout << quality.error_percentage << "\t, ";
    std::cout << quality.ssim << "\t, ";
    std::cout << quality.pnsr << std::endl;

    std::fstream myfile;
    std::string file_path = "./quality.csv";
    myfile.open(file_path.c_str(), std::ios_base::app);
    assert(myfile.is_open());
    myfile << ",," << quality.input_dist_stats.max << ", "
           << quality.input_dist_stats.min << ", "
           << quality.input_dist_stats.mean << ", "
           << quality.input_dist_stats.sdev << ", "
           << quality.input_dist_stats.entropy << ",,"
           << quality.rmse << "\t, "
           << quality.rmse_percentage << "\t, "
           << quality.error_rate << "\t, "
           << quality.error_percentage << "\t, "
           << quality.ssim << "\t, "
           << quality.pnsr << std::endl;

}

/* print quantized histrogram of mats in integar. */
void Quality::print_histogram(float* input){
    cv::Mat mat, b_hist;
    array2mat(mat, input, this->row, this->col);
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange[] = {range};
    bool uniform = true, accumulate=false;
    calcHist( &mat, 1, 0, cv::Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate );
    std::cout << __func__ << ": hist of mat: " << std::endl;
    for(int i = 0 ; i < histSize ; i++){
        std::cout << b_hist.at<float>(i) << " ";
    }
    std::cout << std::endl;
}

void Quality::print_results(bool is_tiling, int verbose){
    Unit total_quality = {
        this->rmse(),
        this->rmse_percentage(),
        this->error_rate(),
        this->error_percentage(),
        this->ssim(),
        this->pnsr(),
        {this->result.input_dist_stats.max,
         this->result.input_dist_stats.min,
         this->result.input_dist_stats.mean,
         this->result.input_dist_stats.sdev,
         this->result.input_dist_stats.entropy}
    };
    std::vector<Unit> tiling_quality;

    if(is_tiling == true){
        for(int i = 0 ; i < this->row_cnt ; i++){
            for(int j = 0 ; j < this->col_cnt ; j++){
                int idx = i * this->col_cnt + j;
                Unit per_quality = {
                    this->rmse(i, j),
                    this->rmse_percentage(i, j),
                    this->error_rate(i, j),
                    this->error_percentage(i, j),
                    this->ssim(i, j),
                    this->pnsr(i, j),
                    {this->result_pars[idx].input_dist_stats.max,
                     this->result_pars[idx].input_dist_stats.min,
                     this->result_pars[idx].input_dist_stats.mean,
                     this->result_pars[idx].input_dist_stats.sdev,
                     this->result_pars[idx].input_dist_stats.entropy}
                };
                tiling_quality.push_back(per_quality);
            }
        }
    }

    int size = 10;

    float baseline_max = FLT_MIN;
    float baseline_min = FLT_MAX;
    float proposed_max = FLT_MIN;
    float proposed_min = FLT_MAX;

    if(verbose){
        std::cout << "baseline result:" << std::endl;
        for(int i = 0 ; i < this->row ; i++){
            for(int j = 0 ; j < this->col ; j++){
                if(i < size && j < size)
                    std::cout << baseline_mat[i*this->ldn+j] << " ";
                if(baseline_mat[i*this->ldn+j] > baseline_max)
                    baseline_max = baseline_mat[i*this->ldn+j];
                if(baseline_mat[i*this->ldn+j] < baseline_min)
                    baseline_min = baseline_mat[i*this->ldn+j];
            }
            if(i < size)
                std::cout << std::endl;
        }
        std::cout << "proposed result:" << std::endl;
        for(int i = 0 ; i < this->row ; i++){
            for(int j = 0 ; j < this->col ; j++){
                if(i < size && j < size)
                    std::cout << target_mat[i*this->ldn+j] << " ";
                if(target_mat[i*this->ldn+j] > proposed_max)
                    proposed_max = target_mat[i*this->ldn+j];
                if(target_mat[i*this->ldn+j] < proposed_min)
                    proposed_min = target_mat[i*this->ldn+j];
            }
            if(i < size)
                std::cout << std::endl;
        }
    }

    std::cout << "baseline_mat max: " << baseline_max << ", "
              << "min: " << baseline_min << std::endl;
    std::cout << "proposed_mat max: " << proposed_max << ", "
              << "min: " << proposed_min << std::endl;

    printf("=============================================\n");
    printf("Quality results(is_tiling?%d)\n", is_tiling);
    printf("=============================================\n");
    std::cout << "total quality: " << std::endl;

    std::fstream myfile;
    std::string file_path = "./quality.csv";
    myfile.open(file_path.c_str(), std::ios_base::app);
    assert(myfile.is_open());
    
    std::cout << "input(max, min, mean, sdev, entropy) | rmse,\trmse_percentage(\%),\terror_rate(\%),\terror_percentage(\%),\tssim,\tpnsr(dB)" << std::endl;
    myfile << "total quality,,,,,,,,,,,," << std::endl;
    myfile << ",,max, min, mean, sdev, entropy,,\t rmse(\%),\terror_rate(\%),\terror_percentage(\%),\tssim,\tpnsr(dB)" << std::endl;
    print_quality(total_quality);

    if(is_tiling == true){
        std::cout << "tiling quality: " << std::endl;
        std::cout << "(i, j) input(max, min, mean, sdev, entropy) | rmse,\trmse_percentage(\%),\terror_rate(\%),\terror_percentage(\%),\tssim,\tpnsr(dB)" << std::endl;
        myfile << "tiling quality" << std::endl;
        myfile << "(i, j), max, min, mean, sdev, entropy,,\t rmse,\trmse_percentage(\%),\terror_rate(\%),\terror_percentage(\%),\tssim,\tpnsr(dB)" << std::endl;
        for(int i = 0 ; i < this->row_cnt  ; i++){
            for(int j = 0 ; j < this->col_cnt  ; j++){
                std::cout << "(" << i << ", " << j << "): ";
                print_quality(tiling_quality[i*this->col_cnt+j]);
            }
        }
    }

    std::cout << __func__ << ": baseline hist.:" << std::endl;
    print_histogram(this->baseline_mat);
    std::cout << __func__ << ": target hist.:" << std::endl;
    print_histogram(this->target_mat);
}

