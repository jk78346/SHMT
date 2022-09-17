#include "quality.h"

Quality::Quality(int m, int n, int ldn, float* x, float* y){
	this->m = m;
	this->n = n;
	this->ldn = ldn;
	this->x = x;
	this->y = y;
}

float Quality::average(int m, int n, int ldn, float* x){

}

float Quality::sdev(int m, int n, int ldn, float* x, float ux){

}

float Quality::covariance(int m, int n, int ldn, float* x, float* y, float ux, float uy){

}

float Quality::rmse(int m, int n, int ldn, float* x, float* y, int verbose){

} 

float Quality::error_rate(int m, int n, int ldn, float* x, float* y, int verbose){

}

float Quality::error_percentage(int m, int n, int ldn, float* x, float* y, int verbose){

}

float Quality::ssim(int m, int n, int ldn, float* x, float* y, int verbose){

}

float Quality::pnsr(int m, int n, int ldn, float* x, float* y, int verbose){

}
