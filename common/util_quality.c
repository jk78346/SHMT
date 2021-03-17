#include "util_quality.h"
#include "math.h"
#include "stdio.h"

float average(int n, unsigned char* x){
    float sum = 0;
    for(int i = 0 ; i < n ; i++){
        sum += (float)x[i];
    }
    return sum / (float)n;
}

float variance(int n, unsigned char* x){
    float sum = 0;
    float avg = average(n, x);
    for(int i = 0 ; i < n ; i++){
        sum += pow((float)x[i] - avg, 2);
    }
    return sum / pow(n, 2);
}

float covariance(int n , unsigned char* x, unsigned char* y){
    float sum = 0;
    float avg_x = average(n, x);
    float avg_y = average(n, y);
    for(int i = 0 ; i < n ; i++){
        sum += ((float)x[i] - avg_x) * ((float)y[i] - avg_y); 
    }
    return sum / pow(n, 2);
}


float SSIM(int draw_w, int draw_h, unsigned char* buf1, unsigned char* buf2){
/* result */
    float ssim = 0;
/* components */
    int n = draw_w * draw_h;
    float ux = average(n, buf1); // average of x
    float uy = average(n, buf2); // average of y
    float vx = variance(n, buf1); // variance of x
    float vy = variance(n, buf2); // variance of y
    float cov = covariance(n, buf1, buf2); // covariance of x and y
    float L = 255.0; // 2^(# of bits)-1
    float k1 = 0.01; // default
    float k2 = 0.03; // default
    float c1 = pow(k1*L, 2);
    float c2 = pow(k2*L, 2);

    ssim = ((2*ux*uy + c1)*(2*cov+c2)) / ((pow(ux, 2)+pow(uy, 2)+c1)*(pow(vx, 2)+pow(vy, 2)+c2));
    printf("SSIM: %f, ux: %f, uy: %f, cov: %f, vx: %f, vy: %f, c1: %f, c2: %f\n", ssim, ux, uy, cov, vx, vy, c1 , c2);
    return ssim;
}
