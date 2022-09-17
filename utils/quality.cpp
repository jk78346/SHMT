#include "quality.h"
#include "math.h"
#include <float.h>

Quality::Quality(int m, int n, int ldn, float* x, float* y){
	this->row          = m;
	this->col          = n;
	this->ldn          = ldn;
	this->target_mat   = x;
	this->baseline_mat = y;
}

void Quality::get_minmax(float* x, float& max, float& min){
	float curr_max = FLT_MIN;
	float curr_min = FLT_MAX;
	for(int i = 0 ; i < this->row ; i++){
		for(int j = 0 ; j <this->col ; j++){
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

float Quality::average(float* x){
	double sum = 0.0;
	for(int i = 0 ; i < this->row ; i++){
		for(int j = 0 ; j <this->col ; j++){
			sum += x[i*this->ldn+j];
		}
	}
	return (float)(sum / (double)(this->row*this->col));
}

float Quality::sdev(float* x){
	double sum = 0;
	float ux = this->average(x);
	for(int i = 0 ; i < this->row ; i++){
		for(int j = 0 ; j < this->col ; j++){
			sum += pow(x[i*this->ldn+j] - ux, 2);
		}
	}
	return pow((float)(sum / (double)(this->row*this->col)), 0.5);
}

float Quality::covariance(){
	double sum = 0;
	float ux = this->average(this->target_mat);
	float uy = this->average(this->baseline_mat);
	for(int i = 0 ; i < this->row ; i++){
		for(int j = 0 ; j < this->col ; j++){
			sum += (this->target_mat[i*this->ldn+j] - ux) * (this->baseline_mat[i*this->ldn+j] - uy);
		}
	}
	return (float)(sum / (double)(this->row*this->col));
		
}

float Quality::rmse(int verbose){
	double mse = 0;
	double mean = this->average(this->baseline_mat);
	int cnt = 0;
	for(int i = 0 ; i < this->row ; i++){
		for(int j = 0 ; j < this->col ; j++){
			int idx = i*this->ldn+j;
			mse = (mse * cnt + pow(this->target_mat[idx] - this->baseline_mat[idx], 2)) / (cnt + 1);
			cnt++;	
		}
	}
	return (sqrt(mse)/mean) * 100.0;
} 

float Quality::error_rate(int verbose){
	double rate = 0;
	double mean = this->average(this->baseline_mat);
	int cnt = 0;
	for(int i = 0 ; i < this->row ; i++){
		for(int j = 0 ; j < this->col ; j++){
			int idx = i*this->ldn+j;
			rate = (rate * cnt + fabs(this->target_mat[idx] - this->baseline_mat[idx])) / (cnt + 1);
			cnt++;	
		}
	}
	return (rate / mean) * 100.0;
}

float Quality::error_percentage(int verbose){
	int cnt = 0;
	for(int i = 0 ; i < this->row ; i++){
		for(int j = 0 ; j < this->col ; j++){
			int idx = i*this->ldn+j;
			if(fabs(this->target_mat[idx] - this->baseline_mat[idx]) < 1e-8){
				cnt++;
			}
		}
	}
	return ((float)cnt / (float)(this->row*this->col)) * 100.0;
}

float Quality::ssim(int verbose){
	// SSIM default parameters
	int    L = 255.0; // 2^(# of bits) - 1
	float k1 = 0.01;
	float k2 = 0.03;
	float c1 = 6.5025;  // (k1*L)^2
	float c2 = 58.5225; // (k2*L)^2
	float target_max, target_min;
	this->get_minmax(this->target_mat, target_max, target_min);
	
	// update dynamic range
	L = fabs(target_max - target_min); 
	c1 = (k1*L)*(k1*L);
	c2 = (k2*L)*(k2*L);

	// main calculation
	float ssim = 0.0;
	
	int len = this->row * this->col;
	float ux = this->average(this->target_mat);
	float uy = this->average(this->baseline_mat);
	float vx = this->sdev(this->target_mat);
	float vy = this->sdev(this->baseline_mat);
	float cov = this->covariance();

	ssim = ((2*ux*uy+c1) * (2*cov+c2)) / ((pow(ux, 2) + pow(uy, 2) + c1) * (pow(vx, 2) + pow(vy, 2) + c2));
	return ssim;
}

float Quality::pnsr(int verbose){
	float baseline_max, baseline_min;
	this->get_minmax(this->baseline_mat, baseline_max, baseline_min);
	double mse = 0;
	double mean = this->average(this->baseline_mat);
	int cnt = 0;
	for(int i = 0 ; i < this->row ; i++){
		for(int j = 0 ; j < this->col ; j++){
			int idx = i*this->ldn+j;
			mse = (mse * cnt + pow(this->target_mat[idx] - this->baseline_mat[idx], 2)) / (cnt + 1);
			cnt++;	
		}
	}
	return 20*log10(baseline_max) - 10*log10(mse/mean);
}
