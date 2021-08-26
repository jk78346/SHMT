#include "util_quality.h"
#include "math.h"
#include "stdio.h"

float average(int n, uint8_t* x){
    float sum = 0;
    for(int i = 0 ; i < n ; i++){
        sum += (float)x[i];
    }
    return sum / (float)n;
}

float variance(int n, uint8_t* x){
    float sum = 0;
    float avg = average(n, x);
    for(int i = 0 ; i < n ; i++){
        sum += pow((float)x[i] - avg, 2);
    }
    return sum / (float)n;
}

float covariance(int n , uint8_t* x, uint8_t* y){
    float sum = 0;
    float avg_x = average(n, x);
    float avg_y = average(n, y);
    for(int i = 0 ; i < n ; i++){
        sum += ((float)x[i] - avg_x) * ((float)y[i] - avg_y); 
    }
    return sum / (float) n;
}


float SSIM(int draw_w, int draw_h, uint8_t* buf1, uint8_t* buf2){
/* result */
    float ssim = 0;
/* components */
    int n = draw_w * draw_h;
    float ux = average(n, buf1); // average of x
    float uy = average(n, buf2); // average of y
    float vx = variance(n, buf1); // variance of x
    float vy = variance(n, buf2); // variance of y
    float cov = covariance(n, buf1, buf2); // covariance of x and y
    float L = 255; // 2^(# of bits)-1
    float k1 = 0.01; // default
    float k2 = 0.03; // default
    float c1 = (k1*L)*(k1*L);
    float c2 = (k2*L)*(k2*L);

    ssim = (2*ux*uy + c1)*(2*cov+c2) / ((pow(ux, 2)+pow(uy, 2)+c1)*(pow(vx, 2)+pow(vy, 2)+c2));
    printf("SSIM: %f\n", ssim);
    return ssim;
}
