////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//
// Matrix multiplication: C = A * B.
// Host code.
//
// This sample implements matrix multiplication as described in Chapter 3
// of the programming guide and uses the CUBLAS library to demonstrate
// the best performance.

// SOME PRECAUTIONS:
// IF WE WANT TO CALCULATE ROW-MAJOR MATRIX MULTIPLY C = A * B,
// WE JUST NEED CALL CUBLAS API IN A REVERSE ORDER: cublasSegemm(B, A)!
// The reason is explained as follows:

// CUBLAS library uses column-major storage, but C/C++ use row-major storage.
// When passing the matrix pointer to CUBLAS, the memory layout alters from
// row-major to column-major, which is equivalent to an implicit transpose.

// In the case of row-major C/C++ matrix A, B, and a simple matrix multiplication
// C = A * B, we can't use the input order like cublasSgemm(A, B)  because of
// implicit transpose. The actual result of cublasSegemm(A, B) is A(T) * B(T).
// If col(A(T)) != row(B(T)), equal to row(A) != col(B), A(T) and B(T) are not
// multipliable. Moreover, even if A(T) and B(T) are multipliable, the result C
// is a column-based cublas matrix, which means C(T) in C/C++, we need extra
// transpose code to convert it to a row-based C/C++ matrix.

// To solve the problem, let's consider our desired result C, a row-major matrix.
// In cublas format, it is C(T) actually (because of the implicit transpose).
// C = A * B, so C(T) = (A * B) (T) = B(T) * A(T). Cublas matrice B(T) and A(T)
// happen to be C/C++ matrice B and A (still because of the implicit transpose)!
// We don't need extra transpose code, we only need alter the input order!
//
// CUBLAS provides high-performance matrix multiplication.
// See also:
// V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
// in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
// Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
//

// Utilities and system includes
#include <assert.h>
//#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
//#include <cuda_runtime.h>
//#include <cublas_v2.h>

// CUDA and CUBLAS functions
//#include <helper_functions.h>
//#include <helper_cuda.h>
/* ============================================= */
#include <chrono>
#include <float.h>
#include <random>
#include <fcntl.h> // for O_RDWR
#include <unistd.h> // for open()
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "gptpu.h"
#include "math.h"
#include "omp.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

Mat* crops;
Mat* out_pars;

#ifndef get_time_ms
#define get_time_ms(_end, _start) (std::chrono::duration_cast<std::chrono::nanoseconds>(_end - _start).count()/1000000.0)
#endif

// use edgeTPU
int     L = 255.0; // 2^(# of bits) - 1
#define k1 0.01 // default
#define k2 0.03 // default
float c1 = 6.5025; // (k1*L)*(k1*L)
float c2 = 58.5225; // (k2*L)*(k2*L)

#define E 0.001 // epsilon

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing;
typedef std::chrono::high_resolution_clock clk;

float mix_p = 0.5; // percentage of weighted RR on GPU if mix mode is used

unsigned int COMMON_BLK = 2048; // default is optimal
unsigned int BLK_H = COMMON_BLK;
unsigned int BLK_W = COMMON_BLK;

int devID = 0; // GPU devID 

using namespace std;

/* ============================================= */

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int IN_W, 
                 IN_H, 
                 IN_BLK_W,
                 IN_BLK_H,
                 IN_C, 
                 F_W, 
                 F_H, 
                 S_W, 
                 S_H, 
                 OUT_C, 
                 OUT_W, 
                 OUT_H, 
                 OUT_BLK_W, 
                 OUT_BLK_H, 
                 OUT_W_BLK_CNT, 
                 OUT_H_BLK_CNT;
} sMatrixSize;

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void
matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}

vector<string> split (string s, string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;
    while ((pos_end = s.find (delimiter, pos_start)) != string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }
    res.push_back (s.substr (pos_start));
    return res;
}

int findSortedIndex(vector<pair<int, int>> &arr, int idx){
        for(auto i = 0 ; i < arr.size() ; ++i){
                if(arr[i].first == idx){
                        return i;
                }
        }
        return -1;
}

int findIndex(vector<string> &arr, string item){
        for(auto i = 0 ; i < arr.size() ; ++i){
                if(arr[i] == item){
                        return i;
                }
        }
        return -1;
}

// Allocates a matrix with random float entries.
void randomInit(float *data, int m, int n, int _mode)
{
    if(_mode == 6){ // Gx Sobel filter weights
        data[0] = -1;
        data[1] =  0;
        data[2] =  1;
        data[3] = -2;
        data[4] =  0;
        data[5] =  2;
        data[6] = -1;
        data[7] =  0;
        data[8] =  1;
        return;
    }

    std::default_random_engine gen;
    std::normal_distribution<float> dis(128.0, 32.0);

    const std::string file_name = "./data/lena_gray_2Kx2K.bmp";
    int fd = open(file_name.c_str(), O_RDONLY);
    char *src;
    struct stat st_temp;
    fstat(fd, &st_temp);
    if(fd < 0){
        std::cout << __func__ << ": image file " << file_name << " opening fail." << std::endl; exit(0);
    }
    src = static_cast<char*>(mmap(NULL, st_temp.st_size, PROT_READ, MAP_SHARED, fd, 0));
    assert(src != MAP_FAILED);

    printf("data generating mode: %d\n", _mode);
    for (int i = 0; i < m; ++i){
        for(int j = 0 ; j < n ; ++j){
                if(_mode == 0){  // Uniform distribution
                        data[i*n+j] = (float)((float)rand() / (float)(RAND_MAX/1));
 
                }else if(_mode == 1){  // Normal distribution
                        data[i*n+j] = dis(gen);

                }else if(_mode == 2){  // Hankel Matirx
                        data[i*n+j] = 1.0/(i+j+1);  // Hilbert matrix, an example of Hankel matrix

                }else if(_mode == 3){ // Frank matrix
                        data[i*n+j] = m*n+1-((i >= j)?i:j);                

                }else if(_mode == 4){  // Read image file
                        data[i*n+j] = src[i*n+j+1080]; // header size of bmp is 1080
                        if(i == 0 && j == 0){
                                printf("read image file: %s\n", file_name.c_str());
                        }
                }else if(_mode == 5){ // read mtx file
                        if(i == 0 && j == 0){
                                const std::string mtx_file = "./data/dw8192.mtx"; //"./data/gemat12.mtx";//"./data/dw8192.mtx"; //"./data/beause.mtx";
                                printf("read mtx filer: %s\n", mtx_file.c_str());
                                openctpu_read_mmio(mtx_file.c_str(), data, m, n, n);
                        }
                }else {
                        printf("data Initialization mode %d undefined, exit\n", _mode);
                        exit(0);
                }
                //if(i < 5 && j < 5){
                //        std::cout << __func__ << ", data[" << i << "*" << n << "+" << j << "]: " << data[i*n+j] << std::endl;
                //}
        }
    }
    munmap(src, st_temp.st_size);
    close(fd);
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;

    for (j = 0; j < height; j++)
    {
        if (error_count < iListLength)
        {
            printf("\n  Row %d:\n", j);
        }

        for (i = 0; i < width; i++)
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol)
            {
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}

void initializeSHAPE(int argc, char **argv, int &iSizeMultiple, sMatrixSize &matrix_size)
{
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line

    int block_size = atoi(argv[2]);  // iSizeMultiple is 5

    matrix_size.IN_W = 4096;
    matrix_size.IN_H = 4096;
    matrix_size.IN_C = 1;
    matrix_size.F_W = 3;
    matrix_size.F_H = 3;
    matrix_size.S_W = 1;
    matrix_size.S_H = 1;
    matrix_size.OUT_C = 1;

    printf("input(IN_W: %u, IN_H: %u, IN_C: %u), filter(F_W: %u, F_H: %u, S_W: %u, S_H: %u, OUT_C: %u)\n",
           matrix_size.IN_W, matrix_size.IN_H, matrix_size.IN_C,
           matrix_size.F_W, matrix_size.F_H, matrix_size.S_W, matrix_size.S_H, matrix_size.OUT_C);

}

float average(int m, int n, int ldn, float* x){
	double sum = 0;
        for(int i = 0 ; i < m ; i++){
	        for(int j = 0 ; j < n ; j++){
		        sum += x[i*ldn+j];
	        }
        }
	return (float)(sum / (double)(m*n));
}

float sdev(int m , int n, int ldn, float* x, float ux){
	double sum = 0;
	float avg = ux;
        for(int i = 0 ; i < m ; i ++){
	        for(int j = 0 ; j < n ; j++){
	        	sum += pow(x[i*ldn+j] - avg, 2);
        	}
        }
	return pow((float)(sum / (double)(m*n)), 0.5);
}

float covariance(int m, int n, int ldn, float* x, float* y, float ux, float uy){
	double sum = 0;
	float avg_x = ux;
	float avg_y = uy;
        for(int i = 0 ; i < m ; i++){
	        for(int j = 0 ; j < n ; j++){
		        sum += (x[i*ldn+j] - avg_x) * (y[i*ldn+j] - avg_y);
	        }
        }
	return (float)(sum / (double)(m*n));
}

float RMSE(int w, int h, int ldn, float* buf1, float* buf2, int verbose){
	double  MSE = 0;
	double mean = 0;
        for(int i = 0; i < w ; i++){
	        for(int j = 0 ; j < h ; j++){
		        MSE  = (MSE * i + pow(buf1[i*ldn+j] - buf2[i*ldn+j], 2)) / (i+1);
		        mean = (mean * i + buf1[i*ldn+j]) / (i+1);
	        }
        }
	return (sqrt(MSE)/mean)*100;
}

float ERROR_RATE(int w, int h, int ldn, float* buf1, float* buf2, int verbose){
	double rate = 0;
	double mean = 0;
        for(int i = 0 ; i < w ; i++){
	        for(int j = 0 ; j < h ; j++){
		        if(verbose > 0){
//                    if(fabs(buf1[i*ldn+j] - buf2[i*ldn+j]) > 1e-5){
//                        std::cout << "buf1[" << i << ", " << j << "] = " << buf1[i*ldn+j] << ", buf2[" << i << ", " << j << "] = " << buf2[i*ldn+j] << std::endl;
//                        exit(0);
//                    }
                }
                mean = (mean * i + buf1[i*ldn+j]) / (i+1);
		        rate = (rate * i + fabs(buf1[i*ldn+j] - buf2[i*ldn+j])) / (i+1); 
	        }
        }
	return (rate/mean)*100;
}

float ERROR_PERCENTAGE(int w, int h, int ldn, float* buf1, float* buf2, int verbose){
	long int  cnt = 0;
        for(int i = 0 ; i < w ; i++){
        	for(int j = 0 ; j < h ; j++){
        		if((long int)(buf1[i*ldn+j]) != (long int)(buf2[i*ldn+j])){
		        	cnt++;
	        	}
        	}
        }
	return ((float)cnt/(float)(w*h))*100;
}

float SSIM(int w, int h, int ldn, float* buf1, float* buf2, int verbose, int casted_to_int){
	if(casted_to_int){
                for(int i = 0 ; i < w ; i++){
		        for(int j = 0 ; j < h ; j++){
			        buf1[i] = std::round(buf1[i*ldn+j]);
		        	buf2[i] = std::round(buf2[i*ldn+j]);
                        }
		}
	}
	float max1 = FLT_MIN;
	float min1 = FLT_MAX;
	float max2 = FLT_MIN;
	float min2 = FLT_MAX;
        for(int i = 0 ; i < w ; i++){
	        for(int j = 0 ; j < h ; j++){
	        	if(buf1[i*ldn+j] > max1){ max1 = buf1[i*ldn+j]; }
	        	if(buf1[i*ldn+j] < min1){ min1 = buf1[i*ldn+j]; }
	        	if(buf2[i*ldn+j] > max2){ max2 = buf2[i*ldn+j]; }
		        if(buf2[i*ldn+j] < min2){ min2 = buf2[i*ldn+j]; }
	        }
        }
	L = fabs(max1 - min1); // update dynamic range 
        c1 = (k1*L)*(k1*L);
        c2 = (k2*L)*(k2*L);
/* verbose */
	if(verbose > 0){
		printf("output casted to int? %d\n", casted_to_int);
		printf("h_baseline: \n");
		for(int i = 0 ; i < 5 ; i++){
			for(int j = 0 ; j < 5 ; j++){
				std::cout << std::fixed << buf1[i*ldn+j] << " ";
			}
			printf("\n");
		}
		printf("h_proposed: \n");
		for(int i = 0 ; i < 5 ; i++){
			for(int j = 0 ; j < 5 ; j++){
				std::cout << std::fixed << buf2[i*ldn+j] << " ";
			}
			printf("\n");
		}
		printf("pair-wise proposed/baseline value ratio: \n");
		for(int i = 0 ; i < 5 ; i++){
			for(int j = 0 ; j < 5 ; j++){
				std::cout << std::fixed << buf2[i*h+j]/buf1[i*ldn+j] << " ";
			}
			printf("\n");
		}
	}
/* result */
	float ssim = 0;
/* components */
	int n = w * h;
	float ux; // average of x
	float uy; // average of y
	float vx; // standard variationof x
	float vy; // standard variationof y
	float cov; //covariance of x and y
	ux = average(w, h, ldn, buf1);
	uy = average(w, h, ldn, buf2);
	vx = sdev(w, h, ldn, buf1, ux);
	vy = sdev(w, h, ldn, buf2, uy);
	cov = covariance(w, h, ldn, buf1, buf2, ux, uy);
	ssim = ((2*ux*uy+c1) * (2*cov+c2)) / ((pow(ux, 2) + pow(uy, 2)+c1) * (pow(vx, 2) + pow(vy, 2) +c2));
	return ssim;
}

float PSNR(int w, int h, int ldn, float* buf1, float* buf2, int verbose){
	double  MSE = 0;
	double mean = 0;
	float  max_v = FLT_MIN;
        for(int i = 0 ; i < w ; i++){
        	for(int j = 0 ; j < h ; j++){
        		if(buf2[i*ldn+j] > max_v){ max_v = buf2[i*ldn+j]; }
        		MSE  = (MSE * i + pow(buf1[i*ldn+j] - buf2[i*ldn+j], 2)) / (i+1);
		        mean = (mean * i + buf1[i*ldn+j]) / (i+1);
	        }
        }
	return 20*log10(max_v) - 10*log10(MSE/mean);
}
/* =============================================================================================================== */
void matrix_mul(openctpu_buffer *matrix_a,
		openctpu_buffer	*matrix_b,
		openctpu_buffer	*matrix_c, float coef){
	openctpu_invoke_operator(coef, "mm_model", matrix_a, matrix_b, matrix_c);
}
/* =============================================================================================================== */

void print_matrix(sMatrixSize matrix_size){
    printf("IN_W: %d\n", matrix_size.IN_W);
    printf("IN_H: %d\n", matrix_size.IN_H);
    printf("IN_C: %d\n", matrix_size.IN_C);
    printf("F_W: %d\n", matrix_size.F_W);
    printf("F_H: %d\n", matrix_size.F_H);
    printf("S_W: %d\n", matrix_size.S_W);
    printf("S_H: %d\n", matrix_size.S_H);
    printf("OUT_C: %d\n", matrix_size.OUT_C);
    printf("OUT_W: %d\n", matrix_size.OUT_W);
    printf("OUT_H: %d\n", matrix_size.OUT_H);
    printf("OUT_W_BLK_CNT: %d\n", matrix_size.OUT_W_BLK_CNT);
    printf("OUT_H_BLK_CNT: %d\n", matrix_size.OUT_H_BLK_CNT);
    printf("OUT_BLK_W: %d\n", matrix_size.OUT_BLK_W);
    printf("OUT_BLK_H: %d\n", matrix_size.OUT_BLK_H);
}

void Mat2array(Mat& img, float* data){
    // data has to be pre-allocated with proper size
    if(! img.isContinuous()){
        img = img.clone();
    }
    // row-major
    for(int i = 0 ; i < img.rows ; i++){
        for(int j = 0 ; j < img.cols ; j++){
            int idx = i*(img.cols)+j;
            data[idx] = img.data[idx]; // uint8_t to float conversion
        }
    }
}

//int cnt = 0;

float conv_CPU(int nIter, sMatrixSize matrix_size, const float alpha, Mat& img, float* h_in, float* h_filter, const float beta, Mat& out_img, float* h_C){
	printf("calling conv_CPU...\n");
   
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    
    Sobel(img, grad_x, CV_16S/*ddepth*/, 1, 0, 3/*ksize*/, 1/*scale*/, 0/*delta*/, BORDER_CONSTANT);
    Sobel(img, grad_y, CV_16S/*ddepth*/, 0, 1, 3/*ksize*/, 1/*scale*/, 0/*delta*/, BORDER_CONSTANT);

    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, out_img);

    //printf("grad size: %d, %d\n", grad.size().width, grad.size().height);
    Mat2array(out_img, h_C);

    //imwrite("sobel_"+std::to_string(cnt)+".jpg", grad);
    //cnt+=1;

    return 0;
}

void reverse_crop(sMatrixSize matrix_size, Mat& out_img, float* h_C, float** h_C_partial){
//    for(int i = 0 ; i < matrix_size.OUT_W_BLK_CNT ; i++){
//        for(int j = 0 ; j < matrix_size.OUT_H_BLK_CNT ; j++){
//            int idx = i*(matrix_size.OUT_H_BLK_CNT)+j;
//            for(int w = 0 ; w < matrix_size.OUT_BLK_W ; w++){
//                for(int h = 0 ; h < matrix_size.OUT_BLK_H ; h++){
//                    int offset = w*(matrix_size.OUT_BLK_H)+h;
//                    h_C[(i*matrix_size.OUT_BLK_W+w) * matrix_size.OUT_H + (j*matrix_size.OUT_BLK_H+h)] = h_C_partial[idx][offset];
//                }
//            }
//        }
//    }
    for(int i = 0 ; i < matrix_size.OUT_W_BLK_CNT ; i++){
        for(int j = 0 ; j < matrix_size.OUT_H_BLK_CNT ; j++){
            out_pars[i*matrix_size.OUT_H_BLK_CNT+j].copyTo(out_img(Rect(i*matrix_size.OUT_BLK_W, j*matrix_size.OUT_BLK_H, matrix_size.OUT_BLK_W, matrix_size.OUT_BLK_H)));
        }
    }    
}

float conv_CPU_TILES(int nIter, sMatrixSize matrix_size, const float alpha, float** h_partial_in, float* h_filter, const float beta, Mat& out_img, float* h_C, float** h_C_partial){
	printf("calling conv_CPU_TILES...\n");
    for(int i = 0 ; i < matrix_size.OUT_W_BLK_CNT ; i++){
        for(int j = 0 ; j < matrix_size.OUT_H_BLK_CNT ; j++){
            int idx = i*(matrix_size.OUT_H_BLK_CNT)+j;
            conv_CPU(nIter, matrix_size, alpha, crops[idx], h_partial_in[idx], h_filter, beta, out_pars[idx], h_C_partial[idx]);
        }
    }
    // combine partial to one
    reverse_crop(matrix_size, out_img, h_C, h_C_partial);
    Mat2array(out_img, h_C);
    return 0;
}

float conv_TPU(int nIter, int argc, char** argv, sMatrixSize matrix_size, float* h_in, float* h_filter, float* h_TPU){
	printf("calling conv_TPU...\n");
    
    int in_size  = matrix_size.IN_BLK_W * matrix_size.IN_BLK_H;
    int out_size = matrix_size.OUT_BLK_W * matrix_size.OUT_BLK_H;
    int* in  = (int*) malloc(in_size * sizeof(int));
    int* out = (int*) calloc(out_size, sizeof(int));
    for(int i = 0 ; i < in_size ; i++){
        in[i] = h_in[i];
    }
    std::string model_path = "conv_IN_2K_2K_1_F_3_3_1_S_1_1_SAME_Sobel_edgetpu.tflite";
    //run_a_model(model_path, nIter, in , in_size, out, out_size, 255);
    
    for(int i = 0 ; i < out_size ; i++){
        h_TPU[i] = out[i];
    }
}

void ChooseQuantizationParams(float max, float min, double& scale, int& mean){
        const float qmin = 0;
        const float qmax = 255;
        scale = (max - min)/(qmax - qmin);
        const double initial_zero_point = qmin - min / scale;
        std::uint8_t nudged_zero_point = 0;
        if(initial_zero_point < qmin){
                nudged_zero_point = qmin;
        }else if(initial_zero_point > qmax){
                nudged_zero_point = qmax;
        }else{
                nudged_zero_point = static_cast<std::uint8_t>(std::round(initial_zero_point));
        }
        mean = (int)nudged_zero_point;
}

float get_dist_similarity(float* in, int m, int n, int ldn){
        float max = FLT_MIN;
        float min = FLT_MAX;
        double _scale;
        int _mean;
#pragma omp parallel for num_threads(4)
        for(int i = 0 ; i < m ; i++){
                for(int j = 0 ; j < n ; j++){
                        if(in[i*(ldn)+j] > max){ max = in[i*(ldn)+j]; }
                        if(in[i*(ldn)+j] < min){ min = in[i*(ldn)+j]; }
                }
        }
        ChooseQuantizationParams(max, min, _scale, _mean);
        float* dist = (float*) calloc(256, sizeof(float));
        for(int i = 0 ; i < m ; i++){
                for(int j = 0 ; j < n ; j++){
                        dist[(unsigned char)lrint(_mean + in[i*(ldn)+j] / _scale)]++;
                }
        }
        float entropy = 0.0;
        float p;
#pragma omp parallel for num_threads(4)
        for(int i = 0 ; i < 256 ; i++){
                p = dist[i] / (m*n);
                if(p > 0){
                        entropy += p*log(p);
                }
        }
        free(dist);
        return entropy*(-1);
}

float conv_TPU_TILES(int nIter, int argc, char** argv, sMatrixSize matrix_size, float* h_in, float* h_filter, float* h_TPU, float** h_partial_c){
	printf("calling conv_TPU_TILES...\n");
    for(int i = 0 ; i < matrix_size.OUT_W_BLK_CNT ; i++){
        for(int j = 0 ; j < matrix_size.OUT_H_BLK_CNT ; j++){
            int idx = i*(matrix_size.OUT_H_BLK_CNT)+j;
            //conv_CPU(nIter, matrix_size, alpha, crops[idx], h_partial_in[idx], h_filter, beta, h_C_partial[idx]);
        }
    }
    // combine partial to one
    //reverse_crop(matrix_size, h_C, h_C_partial);
    return 0;
	
//	int m     = matrix_size.uiHA;
//	int n     = matrix_size.uiHB;
//	int k     = matrix_size.uiWC;
//	
//	int m_blk_cnt = (m / BLK_M);// + (m % BLK_M != 0)?1:0;
//	int n_blk_cnt = (n / BLK_N);// + (n % BLK_N != 0)?1:0;
//	int k_blk_cnt = (k / BLK_K);// + (k % BLK_K != 0)?1:0;
//
//        // edgeTPU setup
//        openctpu_init(1, 1);
//	openctpu_dimension *matrix_a_d, *matrix_b_d, *matrix_c_d;
//	openctpu_buffer    **tensor_a,  **tensor_b,  *tensor_c, ***tensor_partial_c;
//
//	tensor_a            = (openctpu_buffer**)  malloc(m_blk_cnt * n_blk_cnt * sizeof(openctpu_buffer*));
//	tensor_b            = (openctpu_buffer**)  malloc(n_blk_cnt * k_blk_cnt * sizeof(openctpu_buffer*));
//	tensor_partial_c    = (openctpu_buffer***) malloc(n_blk_cnt * sizeof(openctpu_buffer**));
//	//float** h_partial_c = (float**) malloc(n_blk_cnt * sizeof(float*));
//
//	for(int i = 0 ; i < n_blk_cnt ; i++){
//		tensor_partial_c[i] = (openctpu_buffer**) malloc(m_blk_cnt * k_blk_cnt * sizeof(openctpu_buffer*));
//		//h_partial_c[i] = (float*) malloc(m * k * sizeof(float)); 
//	}
//
//	timing b_s = clk::now();
//	matrix_a_d = openctpu_alloc_dimension(3, BLK_M, BLK_N, n/*ldm*/);
//	matrix_b_d = openctpu_alloc_dimension(3, BLK_N, BLK_K, k/*ldm*/);
//	matrix_c_d = openctpu_alloc_dimension(3, BLK_M, BLK_K, k/*ldm*/);
//    
//	auto config = openctpu_setConfig(1/*0: int, 1:float*/, false/*exact_mode*/, false/*mm256_mode*/, 1/*chunk_num*/);
//	// These buffers need to know their shape beforehand for re-formating ( for underlying mm2conv)
//  
//        float* tensor_a_average = (float*) malloc(m_blk_cnt * n_blk_cnt * sizeof(float));
//        float* tensor_a_sdev    = (float*) malloc(m_blk_cnt * n_blk_cnt * sizeof(float));
//        float* tensor_b_average = (float*) malloc(m_blk_cnt * n_blk_cnt * sizeof(float));
//        float* tensor_b_sdev    = (float*) malloc(m_blk_cnt * n_blk_cnt * sizeof(float));
//       
//// getting sdev and average of each partitions at the same time
//        timing b_a_s = clk::now();
//	for(int _i = 0 ; _i < m_blk_cnt ; _i++){
//		for(int _j = 0 ; _j < n_blk_cnt ; _j++){
////                        std::cout << "tensor_a[" << _i << ", " << _j << "] entropy: " << get_dist_similarity(&h_A[(_i*BLK_M)*n+(_j*BLK_N)], BLK_M, BLK_N, n) << std::endl;
//			tensor_a[_i*n_blk_cnt+_j] = 
//			  openctpu_create_buffer(argc, argv, matrix_a_d, &h_A[(_i*BLK_M)*n+(_j*BLK_N)], config, false/*b_major*/, 0/*tensor_type*/, 
//                                                 tensor_a_average[_i*n_blk_cnt+_j], tensor_a_sdev[_i*n_blk_cnt+_j]);
//		}
//	}
//	timing b_a_e = clk::now();
//	timing b_b_s = clk::now();
//	for(int _j = 0 ; _j < n_blk_cnt ; _j++){
//		for(int _k = 0 ; _k < k_blk_cnt ; _k++){
////                        std::cout << "tensor_b[" << _j << ", " << _k << "] entropy: " << get_dist_similarity(&h_B[(_j*BLK_N)*k+(_k*BLK_K)], BLK_N, BLK_K, k) << std::endl;
//			tensor_b[_j*k_blk_cnt+_k] = 
//			  openctpu_create_buffer(argc, argv, matrix_b_d, &h_B[(_j*BLK_N)*k+(_k*BLK_K)], config, false/*b_major*/, 1/*tensor_type*/,
//                                                 tensor_b_average[_j*k_blk_cnt+_k], tensor_b_sdev[_j*k_blk_cnt+_k]);
//		}
//	}
//	
//        timing b_b_e = clk::now();
//	timing b_c_s = clk::now();
//        float c_tmp1, c_tmp2;
//	for(int _i = 0 ; _i < m_blk_cnt ; _i++){
//		for(int _j = 0 ; _j < n_blk_cnt ; _j++){
//			for(int _k = 0 ; _k < k_blk_cnt ; _k++){
//				tensor_partial_c[_j][_i*k_blk_cnt+_k] = 
//	      openctpu_create_buffer(argc, argv, matrix_c_d, &h_partial_c[_j][(_i*BLK_M)*k+(_k*BLK_K)], config, false/*b_major*/, 2/*tensor_type*/,
//                                     c_tmp1, c_tmp2); // dummy stats
//			}
//		}
//	}
//	timing b_c_e = clk::now();
//	timing b_e = clk::now();
//        double bms = get_time_ms(b_e, b_s);
//	printf("binary creation time: %f (ms), a: %f, b: %f, c: %f\n", bms, get_time_ms(b_a_e, b_a_s), get_time_ms(b_b_e, b_b_s), get_time_ms(b_c_e, b_c_s));
//
//// show stats (average, sdev) of input tensor(s)
////	for(int _i = 0 ; _i < m_blk_cnt ; _i++){
////		for(int _j = 0 ; _j < n_blk_cnt ; _j++){
////                        int idx = _i*n_blk_cnt+_j; 
////                        printf("tensor_a[%d, %d] average: %f, sdev: %f, cov: %f\n", _i, _j, tensor_a_average[idx], 
////                                                                                            tensor_a_sdev[idx],
////                                                                                            (float)(tensor_a_sdev[idx] / tensor_a_average[idx]));
////                }
////        }
////	for(int _j = 0 ; _j < n_blk_cnt ; _j++){
////		for(int _k = 0 ; _k < k_blk_cnt ; _k++){
////                        int idx = _j*k_blk_cnt+_k;
////                        printf("tensor_b[%d, %d] average: %f, sdev: %f, cov: %f\n", _j, _k, tensor_b_average[idx], tensor_b_sdev[idx],
////                                                                                            (float)(tensor_b_sdev[idx] / tensor_b_average[idx]));
////	        }
////        }
//
//        timing _start = clk::now();	
//	for (int iter = 0; iter < nIter; iter++){
//		for(int _i = 0 ; _i < m_blk_cnt ; _i++){
//			for(int _j = 0 ; _j < n_blk_cnt ; _j++){
//				for(int _k = 0 ; _k < k_blk_cnt ; _k++){
//					openctpu_enqueue(matrix_mul/*kernel name*/, tensor_a[_i*n_blk_cnt+_j], 
//							       	         	    tensor_b[_j*k_blk_cnt+_k], 
//							            	            tensor_partial_c[_j][_i*k_blk_cnt+_k],
//                                                                                    atof(argv[4]));
//				}
//			}
//		}
//	}
//	openctpu_sync(); 
//	openctpu_clean_up();
//	timing _end = clk::now();	
//// summation1
//	float sum = 0.0;
//	int threshold = 10;
//	int count = 0;
//	for(int _i = 0 ; _i < m ; _i++){
//		for(int _k = 0 ; _k < k ; _k++){
//			sum = 0.0;
//			for(int j = 0 ; j < n_blk_cnt ; j++){
//				sum += h_partial_c[j][_i*k+_k];
//				//if(h_partial_c[j][_i*k+_k] != 0){  // find bug from this print, no value for later parts in h_partial_c
//				//	std::cout << "h_partial_c[" << j << "][" << _i << "*" << k << "+" << _k << "]: " << h_partial_c[j][_i*k+_k] << std::endl;
//				//	count++;
//				//}
//			}
//			h_TPU[_i*k+_k] = sum;
//		}
//	}
//// clean up
//	for(int i = 0 ; i < n_blk_cnt ; i++){
//		free(tensor_partial_c[i]);
//		//free(h_partial_c[i]); 
//	}
//	free(tensor_partial_c);
//	//free(h_partial_c); 
//	free(tensor_a);
//	free(tensor_b);
//
//        free(tensor_a_average);
//        free(tensor_a_sdev);
//        free(tensor_b_average);
//        free(tensor_b_sdev);
//
//	float TPU_ms = get_time_ms(_end, _start);
//	return TPU_ms;
}

bool weighted_RR_on_GPU(int idx){
	if(fabs(mix_p - 0.0) < E){
		return 0;
	}else if(fabs(mix_p - 1.0) < E){
		return 1;
	}else if(fabs(mix_p - 0.5) < E){
		return (idx%2);
	}else if(fabs(mix_p - 0.25) < E){
		return (idx%4 == 0);
	}else if(fabs(mix_p - 0.75) < E){
		return (idx%4!=0);
	}
	return idx%2; // default: fair RR
}

float conv_MIX_TILES(int nIter, int argc, char** argv, sMatrixSize matrix_size, float* h_in, float* h_filter, float* h_C, const float alpha, const float beta){
	printf("calling conv_MIX_TILES...\n");
//	timing b_s = clk::now();
//    
//	cudaDeviceProp deviceProp;
//
//        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
//
//        cublasHandle_t handle;
//
//        checkCudaErrors(cublasCreate(&handle));
//	
//	int m     = matrix_size.uiHA;
//	int n     = matrix_size.uiHB;
//	int k     = matrix_size.uiWC;
//	int m_cnt = matrix_size.uiHA/BLK_M;
// 	int n_cnt = matrix_size.uiHB/BLK_N;
//	int k_cnt = matrix_size.uiWC/BLK_K;
//	printf("blk cnts: (%d, %d, %d) for tiling algorithm.\n", m_cnt, n_cnt, k_cnt);
//	
//	// allocate device memory
//   	float *d_A, *d_B, *d_C;
//	float **d_C_partial = (float**) malloc(n_cnt * sizeof(float*));
//	float **h_C_partial = (float**) malloc(n_cnt * sizeof(float*));
//	
//	unsigned int size_A = m * n;  // matrix_size.uiWA * matrix_size.uiHA;
//    	unsigned int mem_size_A = sizeof(float) * size_A;
//    	unsigned int size_B = n * k ; // matrix_size.uiWB * matrix_size.uiHB;
//    	unsigned int mem_size_B = sizeof(float) * size_B;
//   	unsigned int size_C = m * k ; //matrix_size.uiWC * matrix_size.uiHC;
//        unsigned int mem_size_C = sizeof(float) * size_C;
//
//    	checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
//    	checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
//    	checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
//    	checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
//    	checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));
//
//        // setup execution parameters
//        int block_size = 32;
//        dim3 threads(block_size, block_size);
//        dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);
//    	
//	cudaEvent_t start, stop;
//        // Allocate CUDA events that we'll use for timing
//        checkCudaErrors(cudaEventCreate(&start));
//        checkCudaErrors(cudaEventCreate(&stop));
//
//        // edgeTPU setup
//        openctpu_init(1, 1);
//	openctpu_dimension *matrix_a_d, *matrix_b_d, *matrix_c_d;
//	openctpu_buffer    **tensor_a,   **tensor_b,   *tensor_c, ***tensor_partial_c;
//    
//	auto config = openctpu_setConfig(1/*0: int, 1:float*/, false/*exact_mode*/, false/*mm256_mode*/, 1/*chunk_num*/);
//	
//	tensor_a         = (openctpu_buffer**)  malloc(m_cnt * n_cnt * sizeof(openctpu_buffer*));
//	tensor_b         = (openctpu_buffer**)  malloc(n_cnt * k_cnt * sizeof(openctpu_buffer*));
//	tensor_partial_c = (openctpu_buffer***) malloc(n_cnt * sizeof(openctpu_buffer**));
//	float** h_partial_c = (float**) malloc(n_cnt * sizeof(float*));
//	for(int i = 0 ; i < n_cnt ; i++){
//		// TPU part
//		tensor_partial_c[i] = (openctpu_buffer**) malloc(m_cnt * k_cnt * sizeof(openctpu_buffer*));
//		h_partial_c[i]      = (float*)            malloc(m * k * sizeof(float));
//		// GPU part
//		h_C_partial[i] = (float*) malloc(mem_size_C);
//		checkCudaErrors(cudaMalloc((void **) &d_C_partial[i], mem_size_C));
//	}
//
//	matrix_a_d = openctpu_alloc_dimension(3, BLK_M, BLK_N, n/*ldm*/);
//	matrix_b_d = openctpu_alloc_dimension(3, BLK_N, BLK_K, k/*ldm*/);
//	matrix_c_d = openctpu_alloc_dimension(3, BLK_M, BLK_K, k/*ldm*/);
//        
//        float* tensor_a_average = (float*) malloc(m_cnt * n_cnt * sizeof(float));
//        float* tensor_a_sdev    = (float*) malloc(m_cnt * n_cnt * sizeof(float));
//        float* tensor_b_average = (float*) malloc(m_cnt * n_cnt * sizeof(float));
//        float* tensor_b_sdev    = (float*) malloc(m_cnt * n_cnt * sizeof(float));
//
//	timing b_a_s = clk::now();
//	for(int _i = 0 ; _i < m_cnt ; _i++){
//		for(int _j = 0 ; _j < n_cnt ; _j++){
//			tensor_a[_i*n_cnt+_j] =
//			  openctpu_create_buffer(argc, argv, matrix_a_d, &h_A[(_i*BLK_M)*n+(_j*BLK_N)], config, false/*b_major*/, 0/*tensor_type*/,
//                                                 tensor_a_average[_i*n_cnt+_j], tensor_a_sdev[_i*n_cnt+_j]);
//		}
//	}
//	timing b_a_e = clk::now();
//	timing b_b_s = clk::now();
//	for(int _j = 0 ; _j < n_cnt ; _j++){
//		for(int _k = 0 ; _k < k_cnt ; _k++){
//			tensor_b[_j*k_cnt+_k] =
//			  openctpu_create_buffer(argc, argv, matrix_b_d, &h_B[(_j*BLK_N)*k+(_k*BLK_K)], config, false/*b_major*/, 1/*tensor_type*/,
//                                                 tensor_b_average[_j*k_cnt+_k], tensor_b_sdev[_j*k_cnt+_k]);
//		}
//	}
//	timing b_b_e = clk::now();
//        float c_tmp1, c_tmp2;
//	timing b_c_s = clk::now();
//	for(int _i = 0 ; _i < m_cnt ; _i++){
//		for(int _j = 0 ; _j < n_cnt ; _j++){
//			for(int _k = 0 ; _k < k_cnt ; _k++){
//				tensor_partial_c[_j][_i*k_cnt+_k] =
//	      openctpu_create_buffer(argc, argv, matrix_c_d, &h_partial_c[_j][(_i*BLK_M)*k+(_k*BLK_K)], config, false/*b_major*/, 2/*tensor_type*/, 
//                                        c_tmp1, c_tmp2); // dummy stat
//			}
//		}
//	}
//	timing b_c_e = clk::now();
//	timing b_e = clk::now();
//        double bms = get_time_ms(b_e, b_s);
//	printf("binary creation time: %f (ms), a: %f, b: %f, c: %f\n", bms, get_time_ms(b_a_e, b_a_s), get_time_ms(b_b_e, b_b_s), get_time_ms(b_c_e, b_c_s));
//
////	for(int _i = 0 ; _i < m_cnt ; _i++){
////		for(int _j = 0 ; _j < n_cnt ; _j++){
////                        int idx = _i*n_cnt+_j; 
////                        printf("tensor_a[%d, %d] average: %f, sdev: %f, cov: %f\n", _i, _j, tensor_a_average[idx], 
////                                                                                            tensor_a_sdev[idx],
////                                                                                            (float)(tensor_a_sdev[idx] / tensor_a_average[idx]));
////                }
////        }
////	for(int _j = 0 ; _j < n_cnt ; _j++){
////		for(int _k = 0 ; _k < k_cnt ; _k++){
////                        int idx = _j*k_cnt+_k;
////                        printf("tensor_b[%d, %d] average: %f, sdev: %f, cov: %f\n", _j, _k, tensor_b_average[idx], tensor_b_sdev[idx],
////                                                                                            (float)(tensor_b_sdev[idx] / tensor_b_average[idx]));
////	        }
////        }
//	unsigned int edgeTPU_used = 0;
//	unsigned int idx = 0;
//
//	// check m / blk_m is dividable
//        
//	// Record the start event
//        checkCudaErrors(cudaEventRecord(start, NULL));
//	
//	for (int iter = 0; iter < nIter; iter++)
//        {
//            //note cublas is column primary!
//            //need to transpose the order
//	    //
//	  for(int _i = 0 ; _i < m_cnt ; _i++){
//	  	for(int _k = 0 ; _k < k_cnt; _k++){
//	  		for(int _j = 0 ; _j < n_cnt ; _j++){
//				if(weighted_RR_on_GPU(idx)){ // (weighted) Round-Robin between GPU and TPU
//					checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, BLK_M, BLK_N, BLK_K, 
//						&alpha,
//						&d_B[(_j*BLK_N)*k+(_k*BLK_K)], k, 
//						&d_A[(_i*BLK_M)*n+(_j*BLK_N)], n, 
//						&beta, 
//						&d_C_partial[_j][(_i*BLK_M)*k+(_k*BLK_K)], k));
//				
//				}else{
//					// simulate tiling algorithm in perfromance only
//					openctpu_enqueue(matrix_mul/*kernel name*/, tensor_a[_i * n_cnt + _j], 
//							                            tensor_b[_j * k_cnt + _k], 
//										    tensor_partial_c[_j][_i * k_cnt + _k],
//                                                                                    atof(argv[4]));
//					edgeTPU_used++;
//				}
//				idx++;
//			}
//		}
//	    }
//        }
//
//	// Record the stop event
//        checkCudaErrors(cudaEventRecord(stop, NULL));
//
//        // Wait for the stop event to complete
//        checkCudaErrors(cudaEventSynchronize(stop));
//	
//	// wait for openctpu to complete, if ever been invoked among all iterations.
//	if(edgeTPU_used > 0){
//		openctpu_sync(); 
////		openctpu_clean_up();
//	}
//	
//// TODO: coordinate the output summation, don't do overwritting 
//	// copy result from device to host
//	for(int i = 0 ; i < n_cnt ; i++){
//		checkCudaErrors(cudaMemcpy(h_C_partial[i], d_C_partial[i], mem_size_C, cudaMemcpyDeviceToHost));
//	}
//
////summation
//	std::cout << "summation..." << std::endl;
//	float sum = 0.0;
//	int offset = 0;
//	idx = 0;
//	for(int _i = 0 ; _i < m_cnt ; _i++){
//		for(int _k = 0 ; _k < k_cnt ; _k++){
//			for(int bi = 0 ; bi < BLK_M ; bi++){
//				for(int bk = 0 ; bk < BLK_K ; bk++){
//					sum = 0.0;
//					offset = (_i*BLK_M+bi)*k+(_k*BLK_K+bk);
//					for(int _j = 0 ; _j < n_cnt ; _j++){
//						idx = _i*(n_cnt*k_cnt)+_k*(n_cnt)+_j;
//						if(weighted_RR_on_GPU(idx)){ // (weighted) Round-Robin between GPU and TPU
//							sum += h_C_partial[_j][offset];
//						}else{
//							sum += h_partial_c[_j][offset];
//						}
//					}
//					h_C[offset] = sum;
//				}
//			}
//		}
//	}
//
//// clean up
//	for(int i = 0 ; i < n_cnt ; i++){
//		free(tensor_partial_c[i]);
//		free(h_partial_c[i]); 
//		free(h_C_partial[i]);
//	}
//	free(tensor_partial_c);
//	free(h_partial_c); // TPU
//	free(h_C_partial); // GPU
//	free(tensor_a);
//	free(tensor_b);
//        
//        free(tensor_a_average);
//        free(tensor_a_sdev);
//        free(tensor_b_average);
//        free(tensor_b_sdev);
//
//        // Destroy the handle
//        checkCudaErrors(cublasDestroy(handle));
//
//        checkCudaErrors(cudaFree(d_A));
//        checkCudaErrors(cudaFree(d_B));
//        checkCudaErrors(cudaFree(d_C));
//
//	float msecTotal;
//	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
//	return msecTotal;
}

void assign_blk_size(int argc, char** argv){
    if(argc < 7){
    	printf("block size is undefined, exit\n");
	exit(0);
    }
    COMMON_BLK = atoi(argv[7]);
    BLK_W = COMMON_BLK;
    BLK_H = COMMON_BLK;
    printf("BLK_W: %d, BLK_H: %d\n", BLK_W, BLK_H);
}

void assign_mix_p(int argc, char** argv){
	if(argc < 8){
		printf("p on GPU for mix mode is missing, default is set to 0.5 (fair RR)\n");
	}else{	
		mix_p = atof(argv[8]);
		if(fabs(mix_p - 0.0) < E){
			printf("fall back to edgeTPU kernel only (mode 3)\n");
		}else if(fabs(mix_p - 1.0) < E){
			printf("fall back to GPU kernel only (mode 1)\n");
		}
	}
	printf("weighted RR: %4.1f%% sub-tasks on GPU\n", mix_p*100);
}

void read_img(const std::string file_name, sMatrixSize matrix_size, Mat& img){
    Mat raw = imread(file_name);
    assert(!raw.empty());
    cvtColor(raw, img, COLOR_BGR2GRAY);
    std::cout << "img rows    : " << img.rows          << std::endl;
    std::cout << "img cols    : " << img.cols          << std::endl;
    std::cout << "img width   : " << img.size().width  << std::endl;
    std::cout << "img hieght  : " << img.size().height << std::endl;
    std::cout << "img channels: " << img.channels()    << std::endl;

    assert(img.size().width * img.size().height == matrix_size.IN_W * matrix_size.IN_H);
    return;
}

void get_partitions(sMatrixSize matrix_size, int i, int j, Mat& img, Mat& crop){
    assert(i >= 0 && i < matrix_size.OUT_W_BLK_CNT);
    assert(j >= 0 && j < matrix_size.OUT_H_BLK_CNT);
    Rect roi(i*matrix_size.IN_BLK_W, j*matrix_size.IN_BLK_H, matrix_size.IN_BLK_W, matrix_size.IN_BLK_H);
    //crop = img(Range(i * matrix_size.IN_BLK_W, (i+1) * matrix_size.IN_BLK_W),
    //           Range(j * matrix_size.IN_BLK_H, (j+1) * matrix_size.IN_BLK_H));
    std::cout << roi << std::endl;
    img(roi).copyTo(crop);
}

float run_conv(int _mode, int argc, char** argv, int nIter, sMatrixSize matrix_size, const std::string file_name, float alpha, float beta, Mat& img, float* h_in, float** h_partial_in, float* h_filter, Mat& out_img, float* h_C, float** h_partial_C){
	float kernel_ms = 0;
    if(_mode == 0){ // CPU mode
		kernel_ms = conv_CPU(nIter, matrix_size, alpha, img, h_in, h_filter, beta, out_img, h_C);
    }else if(_mode == 1){ // CPU tiling algorithm mode
		kernel_ms = conv_CPU_TILES(nIter, matrix_size, alpha, h_partial_in, h_filter, beta, out_img, h_C, h_partial_C);
	}else if(_mode == 2){ // TPU mode
        	kernel_ms = conv_TPU(nIter, argc, argv, matrix_size, h_in, h_filter, h_C);
	}else if(_mode == 3){ // TPU tiling algorithm mode
        	kernel_ms = conv_TPU_TILES(nIter, argc, argv, matrix_size, h_in, h_filter, h_C, h_partial_C);
	}else if(_mode == 4){ // mix tiling algorithm mode
		assign_mix_p(argc, argv);
        	kernel_ms = conv_MIX_TILES(nIter, argc, argv, matrix_size, h_in, h_filter, h_C, alpha, beta);
	}else if(_mode == -1){ // skip
		printf("skip, no run\n");
	}else{
		printf("undefined mode: %d, exit...\n", _mode);
		exit(0);
	}
	return kernel_ms;
}

void img2ppm(sMatrixSize matrix_size, float* h_c, const std::string file_name){
    FILE *f = fopen(file_name.c_str(), "wb");
    fprintf(f, "P6\n%i %i 255\n", matrix_size.OUT_W, matrix_size.OUT_H);
    for(int i = 0 ; i < matrix_size.OUT_H ; i++){
        for(int j = 0 ; j < matrix_size.OUT_W ; j++){
            unsigned int index = (i)*(matrix_size.OUT_W)+(j);
            //printf("pixel(%d, %d): %d, = %f\n", i, j, (char)h_c[index], h_c[index]);
            fputc((char)h_c[index], f);
            fputc(0, f);
            fputc(0, f);
        }
    }
    fclose(f);
    return;
}

void output_img_to_file(sMatrixSize matrix_size, float* h_c_baseline, float* h_c_proposed){
    img2ppm(matrix_size, h_c_baseline, "./data/lena_4Kx4K_gray_output_baseline.ppm");
    img2ppm(matrix_size, h_c_proposed, "./data/lena_4Kx4K_gray_output_proposed.ppm");
}
////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
int conv(int argc, char **argv, sMatrixSize &matrix_size)
{
    // set seed for rand()
    srand(2006);

    // allocate host memory for input A and filter B
    unsigned int size_A = matrix_size.IN_W * matrix_size.IN_H * matrix_size.IN_C;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_in = (float *)malloc(mem_size_A);
    unsigned int size_B = matrix_size.F_W * matrix_size.F_H * matrix_size.OUT_C;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_filter = (float *)malloc(mem_size_B);

    // initialize host memory
    assert(matrix_size.IN_C == 1 && matrix_size.OUT_C == 1);
    assert(matrix_size.S_W == 1 && matrix_size.S_H == 1);
    
    const std::string file_name = "./data/lena_gray_4Kx4K.bmp";

    //randomInit(h_in, matrix_size.IN_W, matrix_size.IN_H, atoi(argv[1]));
    randomInit(h_filter, matrix_size.F_W, matrix_size.F_H, 6/*Gx Sobel filter*/);

    // allocate host memory for the result
    // no padding is assumed here.
    unsigned int size_C_W = ceil((matrix_size.IN_W - matrix_size.F_W + 2) / matrix_size.S_W) + 1;
    unsigned int size_C_H = ceil((matrix_size.IN_H - matrix_size.F_H + 2) / matrix_size.S_H) + 1;
    matrix_size.OUT_W = size_C_W;
    matrix_size.OUT_H = size_C_H;
    unsigned int size_c = size_C_W * size_C_H;
    unsigned int mem_size_C = sizeof(float) * size_c;
    float *h_C_baseline = (float *) malloc(mem_size_C);
    float *h_C_proposed = (float *) malloc(mem_size_C);

    // assign partitioning
    assign_blk_size(argc, argv);

    matrix_size.IN_BLK_W = BLK_W;
    matrix_size.IN_BLK_H = BLK_H;

    assert(matrix_size.IN_W % matrix_size.IN_BLK_W == 0);
    assert(matrix_size.IN_H % matrix_size.IN_BLK_H == 0);
    int w_cnt = matrix_size.IN_W / matrix_size.IN_BLK_W;
    int h_cnt = matrix_size.IN_H / matrix_size.IN_BLK_H;
    matrix_size.OUT_W_BLK_CNT = w_cnt;
    matrix_size.OUT_H_BLK_CNT = h_cnt;
    assert(size_C_W % w_cnt == 0);
    assert(size_C_H % h_cnt == 0);
    matrix_size.OUT_BLK_W = size_C_W / w_cnt;
    matrix_size.OUT_BLK_H = size_C_H / h_cnt;
    
    float** h_partial_in         = (float**) malloc( w_cnt * h_cnt * sizeof(float*));
    float** h_C_baseline_partial = (float **) malloc(w_cnt * h_cnt * sizeof(float*));
    float** h_C_proposed_partial = (float **) malloc(w_cnt * h_cnt * sizeof(float*));
    for(int i = 0 ; i < (w_cnt * h_cnt) ; i++){
        h_partial_in[i]         = (float*) malloc(matrix_size.IN_BLK_W  * matrix_size.IN_BLK_H );
	    h_C_baseline_partial[i] = (float*) malloc(matrix_size.OUT_BLK_W * matrix_size.OUT_BLK_H);
	    h_C_proposed_partial[i] = (float*) malloc(matrix_size.OUT_BLK_W * matrix_size.OUT_BLK_H);
    }

    // number of iterations
    int nIter = atoi(argv[3]);

    timing _start, _end;
    double baseline_kernel_ms, proposed_kernel_ms;
    double baseline_total_ms,  proposed_total_ms;
    
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    
    print_matrix(matrix_size);
    // Using opencv to partition input image and populate to h_in and h_partial_in
    Mat img;
    Mat out_img;
    read_img(file_name, matrix_size, img);
    Mat2array(img, h_in);
    
    //crops = (Mat*) malloc(matrix_size.OUT_W_BLK_CNT * matrix_size.OUT_H_BLK_CNT * sizeof(Mat));
    crops = new Mat[matrix_size.OUT_W_BLK_CNT * matrix_size.OUT_H_BLK_CNT];
    out_pars = new Mat[matrix_size.OUT_W_BLK_CNT * matrix_size.OUT_H_BLK_CNT];
    for(int i = 0 ; i < matrix_size.OUT_W_BLK_CNT ; i++){
        for(int j = 0 ; j < matrix_size.OUT_H_BLK_CNT ; j++){
            int idx = i*matrix_size.OUT_H_BLK_CNT+j;
            get_partitions(matrix_size, i, j, img, crops[idx]);
            Mat2array(crops[idx], h_partial_in[idx]);
        }
    }
    int _mode;
    _mode = atoi(argv[5]); // baseline
    _start = clk::now();
    baseline_kernel_ms = run_conv(_mode, argc, argv, nIter, matrix_size, file_name, alpha, beta, img, h_in, h_partial_in, h_filter, out_img, h_C_baseline, h_C_baseline_partial);
    _end = clk::now();
    baseline_total_ms = get_time_ms(_end, _start);
	
    _mode = atoi(argv[6]); // proposed
    _start = clk::now();
    proposed_kernel_ms = run_conv(_mode, argc, argv, nIter, matrix_size, file_name, alpha, beta, img, h_in, h_partial_in, h_filter, out_img, h_C_proposed, h_C_proposed_partial);
    _end = clk::now();
    proposed_total_ms = get_time_ms(_end, _start);

    output_img_to_file(matrix_size, h_C_baseline, h_C_proposed);

    // SSIM section
    float ssim             = SSIM(size_C_W, size_C_H, size_C_H, h_C_baseline, h_C_proposed, 1/*verbose*/, 0/*cast to int?*/);
    float ssim_int         = SSIM(size_C_W, size_C_H, size_C_H, h_C_baseline, h_C_proposed, 1/*verbose*/, 1/*cast to int?*/);
    float rmse             = RMSE(size_C_W, size_C_H, size_C_H, h_C_baseline, h_C_proposed, 1/*verbose*/);
    float psnr             = PSNR(size_C_W, size_C_H, size_C_H, h_C_baseline, h_C_proposed, 1/*verbose*/);
    float error_rate       = ERROR_RATE(      size_C_W, size_C_H, size_C_H, h_C_baseline, h_C_proposed, 1/*verbose*/);
    float error_percentage = ERROR_PERCENTAGE(size_C_W, size_C_H, size_C_H, h_C_baseline, h_C_proposed, 1/*verbose*/);

    // quality section
    printf("==============================================================\n");
    printf("Quality result\n");
    printf("==============================================================\n");
    printf("error %%    is: %f%%\t|\t# of elements in difference / total # of elements\n", error_percentage);
    printf("error rate is: %f%%\t|\tsum(abs(xi-yi))/mean(yi)\n", error_rate);
    printf("RMSE       is: %f%%\t|\tsqrt(sum(abs(xi-yi)^2))/mean(yi)\n", rmse);
    printf("PSNR       is: %fdB\t|\t20*log10(MAX(yi)) - 10*log10(sum(abs(xi-yi)^2)/mean(yi))\n", psnr);
    printf("SSIM       is: %f\t\t|\tnaive version, alpha=beta=gamma=1\n", ssim);
    printf("SSIM_int   is: %f\t\t|\tsame SSIM except elements are casted to integer beforehand\n", ssim_int);

    // timing section
    printf("==============================================================\n");
    printf("Latency result\n");
    printf("==============================================================\n");
    printf("\taverage kernel time\taverage total latency time\t(nIter = %d)\n", nIter);
    printf("baseline  : %12.6f (ms) |  %12.6f (ms)\n", baseline_kernel_ms/nIter, baseline_total_ms/nIter);
    printf("proposed  : %12.6f (ms) |  %12.6f (ms)\n", proposed_kernel_ms/nIter, proposed_total_ms/nIter);

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    if(argc < 7){
    	printf("new usage: %s [input mode] [problem size] [nIter] [scale] [baseline's mode] [mode] [block_size, needed for 1, 3, 4] [p for mix mode: p on GPU]\n", argv[0]);
	printf("mode definition:\n");
	printf("\t-1: skip, no run\n");
	printf("\t0: CPU                   mode\n");
	printf("\t1: CPU tiling algorithm  mode\n");
	printf("\t2: TPU                   mode\n");
	printf("\t3: TPU tiling algorithm  mode\n");
	printf("\t4: mix tiling algorithm  mode (round-robin as default)\n");
        exit(0);
    }
    
    printf("[conv2D] - Starting...\n");

    int sizeMult = 5;
    sMatrixSize matrix_size;

    initializeSHAPE(argc, argv, sizeMult, matrix_size);

    int matrix_result = conv(argc, argv, matrix_size);
            
    return matrix_result;
}
