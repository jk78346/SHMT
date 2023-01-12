#include "quality.h"
#include "math.h"
#include <map>
#include <vector>
#include <float.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>

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

float Quality::covariance(int i_start, int j_start, int row_size, int col_size){
	double sum = 0;
	float ux = this->average(this->target_mat, i_start, j_start, row_size, col_size);
	float uy = this->average(this->baseline_mat, i_start, j_start, row_size, col_size);
	for(int i = i_start ; i < i_start+row_size ; i++){
		for(int j = j_start ; j < j_start+col_size ; j++){
			sum += (this->target_mat[i*this->ldn+j] - ux) * (this->baseline_mat[i*this->ldn+j] - uy);
		}
	}
	return (float)(sum / (double)(row_size*col_size));
}

float Quality::rmse_kernel(int i_start, int j_start, int row_size, int col_size){
    double mse = 0;
	double mean = this->average(this->baseline_mat, 
                                i_start,
                                j_start, 
                                row_size, 
                                col_size);
	int cnt = 0;
	for(int i = i_start ; i < i_start+row_size ; i++){
		for(int j = j_start ; j < j_start+col_size ; j++){
			int idx = i*this->ldn+j;
			mse = (mse * cnt + pow(this->target_mat[idx] - this->baseline_mat[idx], 2)) / (cnt + 1);
			cnt++;	
		}
	}
	return (sqrt(mse)/mean) * 100.0;

}

float Quality::rmse(){
    return this->rmse_kernel(0, 0, this->row, this->col);
}

float Quality::rmse(int i, int j){
    return this->rmse_kernel(i*this->row_blk, 
                             j*this->col_blk,
                             this->row_blk, 
                             this->col_blk);
} 

float Quality::error_rate_kernel(int i_start, int j_start, int row_size, int col_size){
	double rate = 0;
	double mean = this->average(this->baseline_mat, i_start, j_start, row_size, col_size);
	int cnt = 0;
	for(int i = i_start ; i < i_start+row_size ; i++){
		for(int j = j_start ; j < j_start+col_size ; j++){
			int idx = i*this->ldn+j;
			rate = (rate * cnt + fabs(this->target_mat[idx] - this->baseline_mat[idx])) / (cnt + 1);
			cnt++;	
		}
	}
	return (rate / mean) * 100.0;
}

float Quality::error_rate(){
    return this->error_rate_kernel(0, 0, this->row, this->col);
}

float Quality::error_rate(int i, int j){
    return this->error_rate_kernel(i*this->row_blk, 
                                   j*this->col_blk,
                                   this->row_blk, 
                                   this->col_blk);
}

float Quality::error_percentage_kernel(int i_start, int j_start, int row_size, int col_size){
	int cnt = 0;
	for(int i = i_start ; i < i_start+row_size ; i++){
		for(int j = j_start ; j < j_start+col_size ; j++){
			int idx = i*this->ldn+j;
			if(fabs(this->target_mat[idx] - this->baseline_mat[idx]) > 1e-8){
				cnt++;
			}
		}
	}
	return ((float)cnt / (float)(row_size*col_size)) * 100.0;
}

float Quality::error_percentage(){
    return this->error_percentage_kernel(0, 0, this->row, this->col);
}

float Quality::error_percentage(int i, int j){
    return this->error_percentage_kernel(i*this->row_blk,
                                         j*this->col_blk,
                                         this->row_blk,
                                         this->col_blk);
}

float Quality::ssim_kernel(int i_start, int j_start, int row_size, int col_size){
	// SSIM default parameters
	int    L = 255.0; // 2^(# of bits) - 1
	float k1 = 0.01;
	float k2 = 0.03;
	float c1 = 6.5025;  // (k1*L)^2
	float c2 = 58.5225; // (k2*L)^2
	float target_max, target_min;
	this->get_minmax(i_start, 
                     j_start,
                     row_size,
                     col_size,
                     this->target_mat,
                     target_max,
                     target_min);
	
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
	float cov = this->covariance(i_start, j_start, row_size, col_size);

	ssim = ((2*ux*uy+c1) * (2*cov+c2)) / ((pow(ux, 2) + pow(uy, 2) + c1) * (pow(vx, 2) + pow(vy, 2) + c2));
	return ssim;
}

float Quality::ssim(){
    return this->ssim_kernel(0, 0 , this->row, this->col);
}

float Quality::ssim(int i, int j){
    return this->ssim_kernel(i*this->row_blk, 
                             j*this->col_blk,
                             this->row_blk,
                             this->col_blk);
}

float Quality::pnsr_kernel(int i_start, int j_start, int row_size, int col_size){
	float baseline_max, baseline_min;
	this->get_minmax(i_start, 
                     j_start,
                     row_size,
                     col_size,
                     this->baseline_mat,
                     baseline_max,
                     baseline_min);
	double mse = 0;
	double mean = this->average(this->baseline_mat, i_start, j_start, row_size, col_size);
	int cnt = 0;
	for(int i = i_start; i < i_start+row_size ; i++){
		for(int j = j_start ; j < j_start+col_size ; j++){
			int idx = i*this->ldn+j;
			mse = (mse * cnt + pow(this->target_mat[idx] - this->baseline_mat[idx], 2)) / (cnt + 1);
			cnt++;	
		}
	}
	return 20*log10(baseline_max) - 10*log10(mse/mean);
}

float Quality::pnsr(){
    return this->pnsr_kernel(0, 0, this->row, this->col);
}

float Quality::pnsr(int i, int j){
    return this->pnsr_kernel(i*this->row_blk,
                             j*this->col_blk, 
                             this->row_blk,
                             this->col_blk);
}

void Quality::print_quality(Unit quality){
    std::cout << "(" << quality.input_dist_stats.mean << ", ";
    std::cout << quality.input_dist_stats.sdev << ", ";
    std::cout << quality.input_dist_stats.entropy << ") | ";
    std::cout << quality.rmse << "\t, ";
    std::cout << quality.error_rate << "\t, ";
    std::cout << quality.error_percentage << "\t, ";
    std::cout << quality.ssim << "\t, ";
    std::cout << quality.pnsr << std::endl;
}

void Quality::print_results(bool is_tiling, int verbose){
    Unit total_quality = {
        this->rmse(),
        this->error_rate(),
        this->error_percentage(),
        this->ssim(),
        this->pnsr(),
        {this->average(this->input_mat, 
                       0,
                       0, 
                       this->row,
                       this->col),
         this->sdev(this->input_mat,
                    0,
                    0,
                    this->row,
                    this->col),
         this->entropy(this->input_mat,
                       0,
                       0,
                       this->row,
                       this->col)}
    };
    std::vector<Unit> tiling_quality;

    if(is_tiling == true){
        for(int i = 0 ; i < this->row_cnt ; i++){
            for(int j = 0 ; j < this->col_cnt ; j++){
                Unit per_quality = {
                    this->rmse(i, j),
                    this->error_rate(i, j),
                    this->error_percentage(i, j),
                    this->ssim(i, j),
                    this->pnsr(i, j),
                    {this->average(this->input_mat, 
                                   i*this->row_blk,
                                   j*this->col_blk, 
                                   this->row_blk,
                                   this->col_blk),
                     this->sdev(this->input_mat,
                                i*this->row_blk,
                                j*this->col_blk,
                                this->row_blk,
                                this->col_blk),
                     this->entropy(this->input_mat,
                                   i*this->row_blk,
                                   j*this->col_blk,
                                   this->row_blk,
                                   this->col_blk)}
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
    std::cout << "input(mean, sdev, entropy) | rmse(\%),\terror_rate(\%),\terror_percentage(\%),\tssim,\tpnsr(dB)" << std::endl;
    print_quality(total_quality);

    if(is_tiling == true){
        std::cout << "tiling quality: " << std::endl;
        std::cout << "(i, j) input(mean, sdev, entropy) | rmse(\%),\terror_rate(\%),\terror_percentage(\%),\tssim,\tpnsr(dB)" << std::endl;
        for(int i = 0 ; i < this->row_cnt  ; i++){
            for(int j = 0 ; j < this->col_cnt  ; j++){
                std::cout << "(" << i << ", " << j << "): ";
                print_quality(tiling_quality[i*this->col_cnt+j]);
            }
        }
    }
}

