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
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
/* ============================================= */
#include <chrono>
#include "gptpu.h"
#include "math.h"
    
#ifndef get_time_ms
#define get_time_ms(_end, _start) (std::chrono::duration_cast<std::chrono::nanoseconds>(_end - _start).count()/1000000.0)
#endif

// use edgeTPU
#define L 255.0 // 2^(# of bits) - 1
#define k1 0.01 // default
#define k2 0.03 // default
#define c1 6.5025 // (k1*L)*(k1*L)
#define c2 58.5225 // (k2*L)*(k2*L)

#define E 0.001 // epsilon

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing;
typedef std::chrono::high_resolution_clock clk;

float mix_p = 0.5; // percentage of weighted RR on GPU if mix mode is used

unsigned int COMMON_BLK = 1024;
unsigned int BLK_M = COMMON_BLK;
unsigned int BLK_N = COMMON_BLK;
unsigned int BLK_K = COMMON_BLK;

int devID = 0; // GPU devID 

/* ============================================= */

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
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

// Allocates a matrix with random float entries.
void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i){
        //data[i] = rand() / (float)RAND_MAX;
        data[i] = (float)(int)((float)rand() / (float)(RAND_MAX/256));
	//if(i < 10) printf("data[%2d]: %f\n", i, data[i]);
    }
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

void initializeCUDA(int argc, char **argv, int &iSizeMultiple, sMatrixSize &matrix_size)
{
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    cudaError_t error;
    devID = 0;

    devID = findCudaDevice(argc, (const char **)argv);

    if (checkCmdLineFlag(argc, (const char **)argv, "sizemult"))
    {
        iSizeMultiple = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
    }

    iSizeMultiple = min(iSizeMultiple, 10);
    iSizeMultiple = max(iSizeMultiple, 1);

    cudaDeviceProp deviceProp;

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    int block_size = atoi(argv[1]);  // iSizeMultiple is 5

    matrix_size.uiWA = 1 * block_size;// * iSizeMultiple;
    matrix_size.uiHA = 1 * block_size;// * iSizeMultiple;
    matrix_size.uiWB = 1 * block_size;// * iSizeMultiple;
    matrix_size.uiHB = 1 * block_size;// * iSizeMultiple;
    matrix_size.uiWC = 1 * block_size;// * iSizeMultiple;
    matrix_size.uiHC = 1 * block_size;// * iSizeMultiple;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           matrix_size.uiHA, matrix_size.uiWA,
           matrix_size.uiHB, matrix_size.uiWB,
           matrix_size.uiHC, matrix_size.uiWC);

    if( matrix_size.uiWA != matrix_size.uiHB ||
        matrix_size.uiHA != matrix_size.uiHC ||
        matrix_size.uiWB != matrix_size.uiWC)
    {
       printf("ERROR: Matrix sizes do not match!\n");
       exit(-1);
    }
}

float average(int n, float* x){
	long double sum = 0;
	for(int i = 0 ; i < n ; i++){
		sum += x[i];
	}
	return (float)(sum / (long double)n);
}

float sdev(int n, float* x, float ux){
	long double sum = 0;
	float avg = ux;
	for(int i = 0 ; i < n ; i++){
		sum += pow(x[i] - avg, 2);
	}
	return pow((float)(sum / (long double)n), 0.5);
}

float covariance(int n, float* x, float* y, float ux, float uy){
	long double sum = 0;
	float avg_x = ux;
	float avg_y = uy;
	for(int i = 0 ; i < n ; i++){
		sum += (x[i] - avg_x) * (y[i] - avg_y);
	}
	return (float)(sum / (long double)n);
}

float RMSE(int w, int h, float* buf1, float* buf2, int verbose){
	long double  MSE = 0;
	long double rate = 0;
	long double mean = 0;
	for(int i = 0 ; i < (w*h) ; i++){
		MSE  = (MSE * i + pow(buf1[i] - buf2[i], 2)) / (i+1);
		mean = (mean * i + buf1[i]) / (i+1);
		rate = (rate * i + fabs(buf1[i] - buf2[i])) / (i+1); 
	}
	return (sqrt(MSE)/mean)*100;
}

float SSIM(int w, int h, float* buf1, float* buf2, int verbose){
/* verbose */
	if(verbose > 0){
		printf("h_baseline: \n");
		for(int i = 0 ; i < 5 ; i++){
			for(int j = 0 ; j < 5 ; j++){
				printf("%12.3f ", buf1[i*h+j]);
			}
			printf("\n");
		}
		printf("h_proposed: \n");
		for(int i = 0 ; i < 5 ; i++){
			for(int j = 0 ; j < 5 ; j++){
				printf("%12.3f ", buf2[i*h+j]);
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
	ux = average(n, buf1);
	uy = average(n, buf2);
	vx = sdev(n, buf1, ux);
	vy = sdev(n, buf2, uy);
	cov = covariance(n, buf1, buf2, ux, uy);
	ssim = ((2*ux*uy+c1) * (2*cov+c2)) / ((pow(ux, 2) + pow(uy, 2)+c1) * (pow(vx, 2) + pow(vy, 2) +c2));
	return ssim;
}
/* =============================================================================================================== */
void matrix_mul(openctpu_buffer *matrix_a,
		openctpu_buffer	*matrix_b,
		openctpu_buffer	*matrix_c){
	openctpu_invoke_operator("mm_model", matrix_a, matrix_b, matrix_c);
}
/* =============================================================================================================== */

float GEMM_GPU(int nIter, sMatrixSize matrix_size, const float alpha, float* h_B, float* h_A, const float beta, float* h_C){
	printf("calling GEMM_GPU...\n");
    
	cudaDeviceProp deviceProp;

        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
        
	cublasHandle_t handle;

        checkCudaErrors(cublasCreate(&handle));
	
	// allocate device memory
   	float *d_A, *d_B, *d_C;
        unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    	unsigned int mem_size_A = sizeof(float) * size_A;
    	unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    	unsigned int mem_size_B = sizeof(float) * size_B;
   	unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
        unsigned int mem_size_C = sizeof(float) * size_C;

    	checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    	checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    	checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    	checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
    	checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));
        
        // setup execution parameters
        int block_size = 32;
        dim3 threads(block_size, block_size);
        dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);
	
	cudaEvent_t start, stop;
        // Allocate CUDA events that we'll use for timing
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        // Record the start event
        checkCudaErrors(cudaEventRecord(start, NULL));
        for (int j = 0; j < nIter; j++)
        {
            //note cublas is column primary!
            //need to transpose the order
            checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
        }
        // Record the stop event
        checkCudaErrors(cudaEventRecord(stop, NULL));

        // Wait for the stop event to complete
        checkCudaErrors(cudaEventSynchronize(stop));
	
	// copy result from device to host
        checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

        // Destroy the handle
        checkCudaErrors(cublasDestroy(handle));
        
	checkCudaErrors(cudaFree(d_A));
        checkCudaErrors(cudaFree(d_B));
        checkCudaErrors(cudaFree(d_C));

	float msecTotal;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	return msecTotal;
}

float GEMM_GPU_TILES(int nIter, sMatrixSize matrix_size, const float alpha, float* h_B, float* h_A, const float beta, float* h_C){
	printf("calling GEMM_GPU_TILES...\n");
	
	int m     = matrix_size.uiWB;
	int n     = matrix_size.uiHA;
	int k     = matrix_size.uiWA;
	int m_cnt = matrix_size.uiWB/BLK_M;
	int n_cnt = matrix_size.uiHA/BLK_N;
	int k_cnt = matrix_size.uiWA/BLK_K;
	printf("blk cnts: (%d, %d, %d) for tiling algorithm.\n", m_cnt, n_cnt, k_cnt);
    
	cudaDeviceProp deviceProp;

        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

        cublasHandle_t handle;

        checkCudaErrors(cublasCreate(&handle));
	
	// allocate device memory
   	float *d_A, *d_B, *d_C;
	float **d_C_partial = (float**)malloc(n_cnt * sizeof(float*));
	float **h_C_partial = (float**)malloc(n_cnt * sizeof(float*));
	
	unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    	unsigned int mem_size_A = sizeof(float) * size_A;
    	unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    	unsigned int mem_size_B = sizeof(float) * size_B;
   	unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
        unsigned int mem_size_C = sizeof(float) * size_C;

    	checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    	checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    	checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    	checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
    	checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));
    
	// allocate partial C
	for(int i = 0 ; i < n_cnt ; i++){
		h_C_partial[i] = (float*) malloc(mem_size_C);
		checkCudaErrors(cudaMalloc((void **) &d_C_partial[i], mem_size_C));
	}

        // setup execution parameters
        int block_size = 32;
        dim3 threads(block_size, block_size);
        dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);
    	
	cudaEvent_t start, stop;
        // Allocate CUDA events that we'll use for timing
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        // Record the start event
        checkCudaErrors(cudaEventRecord(start, NULL));

	// check m / blk_m is dividable
	for (int iter = 0; iter < nIter; iter++)
        {
            //note cublas is column primary!
            //need to transpose the order
	    for(int _i = 0 ; _i < m_cnt ; _i++){
	    	for(int _j = 0 ; _j < n_cnt ; _j++){
			for(int _k = 0 ; _k < k_cnt; _k++){
				checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, BLK_M, BLK_N, BLK_K, 
						&alpha,
						&d_B[(_j*BLK_N)*k+(_k*BLK_K)], m/*lda*/, 
						&d_A[(_i*BLK_M)*n+(_j*BLK_N)], k/*ldb*/, 
						&beta, 
						// This causes overwritting, no partial sum accumulation
						&d_C_partial[_j][(_i*BLK_M)*k+(_k*BLK_K)], m/*ldc*/));
			}
		}
	    }
        }
	
	// Record the stop event
        checkCudaErrors(cudaEventRecord(stop, NULL));

        // Wait for the stop event to complete
        checkCudaErrors(cudaEventSynchronize(stop));
       
	// copy result from device to host
        for(int i = 0 ; i < n_cnt ; i++){
		checkCudaErrors(cudaMemcpy(h_C_partial[i], d_C_partial[i], mem_size_C, cudaMemcpyDeviceToHost));
	}

	// summation
	float sum = 0.0;
	for(int i = 0 ; i < size_C ; i++){
		sum = 0.0;
		for(int p = 0 ; p < n_cnt ; p++){
			sum += h_C_partial[p][i];
		}
		h_C[i] = sum;
	}

        // Destroy the handle
        checkCudaErrors(cublasDestroy(handle));

        checkCudaErrors(cudaFree(d_A));
        checkCudaErrors(cudaFree(d_B));
        checkCudaErrors(cudaFree(d_C));
	for(int i = 0 ; i < n_cnt ; i++){
        	checkCudaErrors(cudaFree(d_C_partial[i]));
		free(h_C_partial[i]);
	}
        free(d_C_partial);
	free(h_C_partial);

	float msecTotal;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	return msecTotal;
}

float GEMM_TPU(int nIter, int argc, char** argv, sMatrixSize matrix_size, float* h_A, float* h_B, float* h_TPU){
	printf("calling GEMM_TPU...\n");
	
	int m = matrix_size.uiWB;
	int n = matrix_size.uiHA;
	int k = matrix_size.uiWA;
	
        // edgeTPU setup
        openctpu_init(1, 1);
	openctpu_dimension *matrix_a_d, *matrix_b_d, *matrix_c_d;
	openctpu_buffer    *tensor_a,   *tensor_b,   *tensor_c;
	
	timing b_s = clk::now();
	matrix_a_d = openctpu_alloc_dimension(3, m, n, n);
	matrix_b_d = openctpu_alloc_dimension(3, n, k, k);
	matrix_c_d = openctpu_alloc_dimension(3, m, k, k);
    
	auto config = openctpu_setConfig(1/*0: int, 1:float*/, false/*exact_mode*/, false/*mm256_mode*/, 1/*chunk_num*/);

	tensor_a = openctpu_create_buffer(argc, argv, matrix_a_d, h_A,   config, false/*b_major*/, 0/*tensor_type*/);
	tensor_b = openctpu_create_buffer(argc, argv, matrix_b_d, h_B,   config, false/*b_major*/, 1/*tensor_type*/);
	tensor_c = openctpu_create_buffer(argc, argv, matrix_c_d, h_TPU, config, false/*b_major*/, 2/*tensor_type*/);
	timing b_e = clk::now();
        double bms = get_time_ms(b_e, b_s);
	printf("binary creation time: %f (ms)\n", bms);

	timing _start = clk::now();	
	for (int j = 0; j < nIter; j++)
        {
		openctpu_enqueue(matrix_mul/*kernel name*/, tensor_a, tensor_b, tensor_c);
	}
	openctpu_sync(tensor_c); 
	timing _end = clk::now();	
	
	float TPU_ms = get_time_ms(_end, _start);
	return TPU_ms;
}

float GEMM_TPU_TILES(int nIter, int argc, char** argv, sMatrixSize matrix_size, float* h_A, float* h_B, float* h_TPU){
	printf("calling GEMM_TPU_TILES...\n");
	
	int m = matrix_size.uiWB;
	int n = matrix_size.uiHA;
	int k = matrix_size.uiWA;

	int m_blk_cnt = (m / BLK_M);// + (m % BLK_M != 0)?1:0;
	int n_blk_cnt = (n / BLK_N);// + (n % BLK_N != 0)?1:0;
	int k_blk_cnt = (k / BLK_K);// + (k % BLK_K != 0)?1:0;

        // edgeTPU setup
        openctpu_init(1, 1);
	openctpu_dimension *matrix_a_d, *matrix_b_d, *matrix_c_d;
	openctpu_buffer    **tensor_a,  **tensor_b,  *tensor_c, ***tensor_partial_c;
	
	tensor_a            = (openctpu_buffer**)  malloc(m_blk_cnt * n_blk_cnt * sizeof(openctpu_buffer*));
	tensor_b            = (openctpu_buffer**)  malloc(n_blk_cnt * k_blk_cnt * sizeof(openctpu_buffer*));
	tensor_partial_c    = (openctpu_buffer***) malloc(n_blk_cnt * sizeof(openctpu_buffer**));
	float** h_partial_c = (float**) malloc(n_blk_cnt * sizeof(float*));
	for(int i = 0 ; i < n_blk_cnt ; i++){
		tensor_partial_c[i] = (openctpu_buffer**) malloc(m_blk_cnt * k_blk_cnt * sizeof(openctpu_buffer*));
		h_partial_c[i] = (float*) malloc(m*k * sizeof(float)); 
	}

	timing b_s = clk::now();
	matrix_a_d = openctpu_alloc_dimension(3, BLK_M, BLK_N, n/*ldm*/);
	matrix_b_d = openctpu_alloc_dimension(3, BLK_N, BLK_K, k/*ldm*/);
	matrix_c_d = openctpu_alloc_dimension(3, BLK_M, BLK_K, k/*ldm*/);
    
	auto config = openctpu_setConfig(1/*0: int, 1:float*/, false/*exact_mode*/, false/*mm256_mode*/, 1/*chunk_num*/);

	for(int _i = 0 ; _i < m_blk_cnt ; _i++){
		for(int _j = 0 ; _j < n_blk_cnt ; _j++){
			tensor_a[_i*n_blk_cnt+_j] = 
			  openctpu_create_buffer(argc, argv, matrix_a_d, &h_A[(_i*m_blk_cnt)*n+(_j*n_blk_cnt)], config, false/*b_major*/, 0/*tensor_type*/);
		}
	}
	for(int _j = 0 ; _j < n_blk_cnt ; _j++){
		for(int _k = 0 ; _k < k_blk_cnt ; _k++){
			tensor_b[_j*k_blk_cnt+_k] = 
			  openctpu_create_buffer(argc, argv, matrix_b_d, &h_B[(_j*n_blk_cnt)*k+(_k*k_blk_cnt)], config, false/*b_major*/, 1/*tensor_type*/);
		}
	}
	for(int _i = 0 ; _i < m_blk_cnt ; _i++){
		for(int _j = 0 ; _j < n_blk_cnt ; _j++){
			for(int _k = 0 ; _k < k_blk_cnt ; _k++){
				tensor_partial_c[_j][_i*k_blk_cnt+_k] = 
	      openctpu_create_buffer(argc, argv, matrix_c_d, &h_partial_c[_j][(_i*m_blk_cnt)*k+(_k*k_blk_cnt)], config, false/*b_major*/, 2/*tensor_type*/);
			}
		}
	}

	timing b_e = clk::now();
        double bms = get_time_ms(b_e, b_s);
	printf("binary creation time: %f (ms)\n", bms);

	timing _start = clk::now();	
	for (int iter = 0; iter < nIter; iter++){
		for(int _i = 0 ; _i < m_blk_cnt ; _i++){
			for(int _j = 0 ; _j < n_blk_cnt ; _j++){
				for(int _k = 0 ; _k < k_blk_cnt ; _k++){
					openctpu_enqueue(matrix_mul/*kernel name*/, tensor_a[_i*n_blk_cnt+_j], 
										    tensor_b[_j*k_blk_cnt+_k], 
										    tensor_partial_c[_j][_i*k_blk_cnt+_k]);
				}
			}
		}
	}
	openctpu_sync(); 
	openctpu_clean_up();
	timing _end = clk::now();	
// summation
	for(int _i = 0 ; _i < m_blk_cnt ; _i++){
		for(int _j = 0 ; _j < n_blk_cnt ; _j++){
			for(int _k = 0 ; _k < k_blk_cnt ; _k++){
				openctpu_get_output(tensor_partial_c[_j][_i*k_blk_cnt+_k]); // includes dequantization (or internal tiles gathering)
			}
		}
	}
	float sum = 0.0;
	for(int i = 0 ; i < (m*k) ; i++){
		sum = 0.0;
		for(int j = 0 ; j < n_blk_cnt ; j++){
			sum += h_partial_c[j][i];
			if(h_partial_c[j][i] == 0){  // find bug from this prin, no value for later parts in h_partial_c
				std::cout << "h_partial_c[" << j << "][" << i << "] is zero." << std::endl;
			}
		}
		h_TPU[i] = sum;
	}
// clean up
	for(int i = 0 ; i < n_blk_cnt ; i++){
		free(tensor_partial_c[i]);
		free(h_partial_c[i]); 
	}
	free(tensor_partial_c);
	free(h_partial_c); 
	free(tensor_a);
	free(tensor_b);

	float TPU_ms = get_time_ms(_end, _start);
	return TPU_ms;
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


float GEMM_MIX_TILES(int nIter, int argc, char** argv, sMatrixSize matrix_size, float* h_A, float* h_B, float* h_C, const float alpha, const float beta){
	printf("calling GEMM_MIX_TILES...\n");
	timing b_s = clk::now();
    
	cudaDeviceProp deviceProp;

        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

        cublasHandle_t handle;

        checkCudaErrors(cublasCreate(&handle));
	
	// allocate device memory
   	float *d_A, *d_B, *d_C;
        unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    	unsigned int mem_size_A = sizeof(float) * size_A;
    	unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    	unsigned int mem_size_B = sizeof(float) * size_B;
   	unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
        unsigned int mem_size_C = sizeof(float) * size_C;

    	checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    	checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    	checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    	checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
    	checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

        // setup execution parameters
        int block_size = 32;
        dim3 threads(block_size, block_size);
        dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);
    	
	cudaEvent_t start, stop;
        // Allocate CUDA events that we'll use for timing
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

	int m     = matrix_size.uiWB;
	int n     = matrix_size.uiHA;
	int k     = matrix_size.uiWA;
	int m_cnt = matrix_size.uiWB/BLK_M;
	int n_cnt = matrix_size.uiHA/BLK_N;
	int k_cnt = matrix_size.uiWA/BLK_K;
	printf("blk cnts: (%d, %d, %d) for tiling algorithm.\n", m_cnt, n_cnt, k_cnt);

        // edgeTPU setup
        openctpu_init(1, 1);
	openctpu_dimension *matrix_a_d, *matrix_b_d, *matrix_c_d;
	openctpu_buffer    *tensor_a,   *tensor_b,   *tensor_c;
	matrix_a_d = openctpu_alloc_dimension(2, BLK_M, BLK_N);
	matrix_b_d = openctpu_alloc_dimension(2, BLK_N, BLK_K);
	matrix_c_d = openctpu_alloc_dimension(2, BLK_M, BLK_K);
    
	auto config = openctpu_setConfig(1/*0: int, 1:float*/, false/*exact_mode*/, false/*mm256_mode*/, 1/*chunk_num*/);

	tensor_a = openctpu_create_buffer(argc, argv, matrix_a_d, h_A,   config, false/*b_major*/, 0/*tensor_type*/);
	tensor_b = openctpu_create_buffer(argc, argv, matrix_b_d, h_B,   config, true /*b_major*/, 1/*tensor_type*/);
	tensor_c = openctpu_create_buffer(argc, argv, matrix_c_d, h_C,   config, false/*b_major*/, 2/*tensor_type*/);
	timing b_e = clk::now();
        double bms = get_time_ms(b_e, b_s);
	printf("binary creation time: %f (ms)\n", bms);

	unsigned int edgeTPU_used = 0;
	unsigned int idx = 0;

	// check m / blk_m is dividable
        
	// Record the start event
        checkCudaErrors(cudaEventRecord(start, NULL));
	
	for (int iter = 0; iter < nIter; iter++)
        {
            //note cublas is column primary!
            //need to transpose the order
	    //
	    for(int _i = 0 ; _i < m_cnt ; _i++){
	    	for(int _j = 0 ; _j < n_cnt ; _j++){
			for(int _k = 0 ; _k < k_cnt; _k++){
				if(weighted_RR_on_GPU(idx)){ // (weighted) Round-Robin between GPU and TPU
					checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, BLK_M, BLK_N, BLK_K, 
						&alpha,
						&d_B[(_j*BLK_N)*k+(_k*BLK_K)], m, 
						&d_A[(_i*BLK_M)*n+(_j*BLK_N)], k, 
						&beta, 
						&d_C[(_i*BLK_M)*k+(_k*BLK_K)], m));
				
				}else{
					// simulate tiling algorithm in perfromance only
					openctpu_enqueue(matrix_mul/*kernel name*/, tensor_a, tensor_b, tensor_c);
					edgeTPU_used++;
				}
				idx++;
			}
		}
	    }
        }

	// Record the stop event
        checkCudaErrors(cudaEventRecord(stop, NULL));

        // Wait for the stop event to complete
        checkCudaErrors(cudaEventSynchronize(stop));
	
	// wait for openctpu to complete, if ever been invoked among all iterations.
	if(edgeTPU_used > 0){
		openctpu_sync(tensor_c); 
		openctpu_clean_up();
	}
	
	// copy result from device to host
        checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
	
        // Destroy the handle
        checkCudaErrors(cublasDestroy(handle));

        checkCudaErrors(cudaFree(d_A));
        checkCudaErrors(cudaFree(d_B));
        checkCudaErrors(cudaFree(d_C));

	float msecTotal;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	return msecTotal;
}

void assign_blk_size(int argc, char** argv){
    if(argc < 5){
    	printf("block size is undefined, exit\n");
	exit(0);
    }
    COMMON_BLK = atoi(argv[5]);
    BLK_M = COMMON_BLK;
    BLK_N = COMMON_BLK;
    BLK_K = COMMON_BLK;

}

void assign_mix_p(int argc, char** argv){
	if(argc < 6){
		printf("p on GPU for mix mode is missing, default is set to 0.5 (fair RR)\n");
	}else{	
		mix_p = atof(argv[6]);
		if(fabs(mix_p - 0.0) < E){
			printf("fall back to edgeTPU kernel only (mode 3)\n");
		}else if(fabs(mix_p - 1.0) < E){
			printf("fall back to GPU kernel only (mode 1)\n");
		}
	}
	printf("weighted RR: %4.1f%% sub-tasks on GPU\n", mix_p*100);
}

float run_GEMM(int _mode, int argc, char** argv, int nIter, sMatrixSize matrix_size, const float alpha, const float beta, float* h_A, float* h_B, float* h_C){
	float kernel_ms = 0;
	if(_mode == 0){ // GPU mode
		kernel_ms = GEMM_GPU(nIter, matrix_size, alpha, h_B, h_A, beta, h_C);
	}else if(_mode == 1){ // GPU tiling algorithm mode
		assign_blk_size(argc, argv);
		kernel_ms = GEMM_GPU_TILES(nIter, matrix_size, alpha, h_B, h_A, beta, h_C);
	}else if(_mode == 2){ // TPU mode
        	kernel_ms = GEMM_TPU(nIter, argc, argv, matrix_size, h_A, h_B, h_C);
	}else if(_mode == 3){ // TPU tiling algorithm mode
		assign_blk_size(argc, argv);
        	kernel_ms = GEMM_TPU_TILES(nIter, argc, argv, matrix_size, h_A, h_B, h_C);
	}else if(_mode == 4){ // mix tiling algorithm mode
		assign_blk_size(argc, argv);
		assign_mix_p(argc, argv);
        	kernel_ms = GEMM_MIX_TILES(nIter, argc, argv, matrix_size, h_A, h_B, h_C, alpha, beta);
	}else if(_mode == -1){ // skip
		printf("skip, no run\n");
	}else{
		printf("undefined mode: %d, exit...\n", _mode);
		exit(0);
	}
	return kernel_ms;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
int matrixMultiply(int argc, char **argv, sMatrixSize &matrix_size)
{
    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // set seed for rand()
    srand(2006);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // allocate host memory for the result
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float *h_C_baseline = (float *) malloc(mem_size_C);
    float *h_C_proposed = (float *) malloc(mem_size_C);

    // number of iterations
    int nIter = atoi(argv[2]);

    timing _start, _end;
    double baseline_kernel_ms, proposed_kernel_ms;
    double baseline_total_ms,  proposed_total_ms;
    
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    int _mode;
    _mode = atoi(argv[3]); // baseline
    _start = clk::now();
    baseline_kernel_ms = run_GEMM(_mode, argc, argv, nIter, matrix_size, alpha, beta, h_A, h_B, h_C_baseline);
    _end = clk::now();
    baseline_total_ms = get_time_ms(_end, _start);
	
    _mode = atoi(argv[4]); // proposed
    _start = clk::now();
    proposed_kernel_ms = run_GEMM(_mode, argc, argv, nIter, matrix_size, alpha, beta, h_A, h_B, h_C_proposed);
    _end = clk::now();
    proposed_total_ms = get_time_ms(_end, _start);

    // SSIM section
    float ssim = SSIM(matrix_size.uiWC, matrix_size.uiHC, h_C_baseline, h_C_proposed, 1/*verbose*/);
    float rmse = RMSE(matrix_size.uiWC, matrix_size.uiHC, h_C_baseline, h_C_proposed, 1/*verbose*/);

    printf("RMSE is: %f%%\n", rmse);
    printf("SSIM is: %f  \n", ssim);

    // timing section
    printf("\taverage kernel time\taverage total latency time\t(nIter = %d)\n", nIter);
    printf("==============================================================\n");
    printf("baseline  : %12.6f (ms) |  %12.6f (ms)\n", baseline_kernel_ms/nIter, baseline_total_ms/nIter);
    printf("proposed  : %12.6f (ms) |  %12.6f (ms)\n", proposed_kernel_ms/nIter, proposed_total_ms/nIter);

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C_baseline);
    free(h_C_proposed);
	printf("done free\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    if(argc < 6){
    	printf("new usage: %s [problem size] [nIter] [baseline's mode] [mode] [block_size, needed for 1, 3, 4] [p for mix mode: p on GPU]\n", argv[0]);
	printf("mode definition:\n");
	printf("\t-1: skip, no run\n");
	printf("\t0: GPU                   mode\n");
	printf("\t1: GPU tiling algorithm  mode\n");
	printf("\t2: TPU                   mode\n");
	printf("\t3: TPU tiling algorithm  mode\n");
	printf("\t4: mix tiling algorithm  mode (round-robin as default)\n");
        exit(0);
    }
    
    printf("[Matrix Multiply CUBLAS] - Starting...\n");

    int sizeMult = 5;
    sMatrixSize matrix_size;

    initializeCUDA(argc, argv, sizeMult, matrix_size);

    int matrix_result = matrixMultiply(argc, argv, matrix_size);
	printf("done result\n");
    return matrix_result;
}
