#include "util_quality.h"
#include "math.h"
#include "stdio.h"

#define L   255.0     // 2^(# of bits)-1
#define k1    0.01    // default
#define k2    0.03    // default
#define c1    6.5025  //(k1*L)*(k1*L)
#define c2   58.5225  //(k2*L)*(k2*L)

float average(int n, unsigned char* x, int type/*RBGA*/){
    float sum = 0;
    for(int i = 0 ; i < n ; i++){
        sum += (float)x[4*i+type];
    }
    return sum / (float)n;
}

float variance(int n, unsigned char* x, float ux, int type/*RBGA*/){
    float sum = 0;
    float avg = ux;
    for(int i = 0 ; i < n ; i++){
        sum += pow((float)x[4*i+type] - avg, 2);
    }
    return sum / (float)n;
}

float covariance(int n , unsigned char* x, unsigned char* y, float ux, float uy, int type/*RBGA*/){
    float sum = 0;
    float avg_x = ux;
    float avg_y = uy;
    for(int i = 0 ; i < n ; i++){
        sum += ((float)x[4*i+type] - avg_x) * ((float)y[4*i+type] - avg_y); 
    }
    return sum / (float)n;
}

float SSIM(int draw_w, int draw_h, unsigned char* buf1, unsigned char* buf2){
/* result */
    float ssim = 0;
    float ssim_per_channel = 0;
/* components */

    int n = draw_w * draw_h;
    float ux; // average of x
    float uy; // average of y
    float vx; // variance of x
    float vy; // variance of y
    float cov; // covariance of x and y

/*rgba order for each pixel*/
    for(int i = 0 ; i < 3 ; i++){ // loop through RGB and skip A
        ux = average(n, buf1, i);
        uy = average(n, buf2, i);
	vx = variance(n, buf1, ux, i);
	vy = variance(n, buf2, uy, i);
	cov = covariance(n, buf1, buf2, ux, uy, i);
        ssim_per_channel = ((2*ux*uy + c1)*(2*cov+c2)) / ((pow(ux, 2)+pow(uy, 2)+c1)*(pow(vx, 2)+pow(vy, 2)+c2));
	printf("SSIM: %f, ux: %f, uy: %f, vx: %f, vy: %f, cov: %f\n", ssim_per_channel, ux, uy, vx, vy, cov);
	ssim += ssim_per_channel; 
    }
    ssim = ssim / 3.0;
    printf("ssim (average over RGB): %f\n", ssim);
    return ssim ;
}
