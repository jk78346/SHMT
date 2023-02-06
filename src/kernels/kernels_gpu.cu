#include <math.h>
#include <string>
#include <stdio.h>
#include "kernels_gpu.h"

const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

static double CND(double d)
{   
    const double       A1 = 0.31938153;
    const double       A2 = -0.356563782;
    const double       A3 = 1.781477937;
    const double       A4 = -1.821255978;
    const double       A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;
    
    double
    K = 1.0 / (1.0 + 0.2316419 * fabs(d));
    
    double
    cnd = RSQRT2PI * exp(- 0.5 * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    
    if (d > 0)
        cnd = 1.0 - cnd;
    
    return cnd;
}

static void BlackScholesBodyCPU(
    float &callResult,
    float &putResult,
    float Sf, //Stock price
    float Xf, //Option strike
    float Tf, //Option years
    float Rf, //Riskless rate
    float Vf  //Volatility rate
)
{   
    double S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;
    
    double sqrtT = sqrt(T);
    double    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);   
    double    d2 = d1 - V * sqrtT;
    double CNDD1 = CND(d1);
    double CNDD2 = CND(d2);
    
    //Calculate Call and Put simultaneously
    double expRT = exp(- R * T);
    callResult   = (float)(S * CNDD1 - X * expRT * CNDD2);
    putResult    = (float)(X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
}

/*
    GPU blackscholes
    Reference: samples/
*/
void GpuKernel::blackscholes_2d(KernelParams& kernel_params, void** in_img, void** out_img){
}


#include <math.h>
#include <string>
#include <stdio.h>
#include "kernels_gpu.h"
#include "BmpUtil.h"
#include "dct8x8_kernel2.cuh"
#include "dct8x8_kernel_quantization.cuh"

#define BENCHMARK_SIZE 10
#define BLOCK_SIZE 8
#define BLOCK_SIZE2 64
#define BLOCK_SIZE_LOG2 3

//float C_a = 1.387039845322148f; //!< a = (2^0.5) * cos(    pi / 16);  Used in forward and inverse DCT.
//float C_b = 1.306562964876377f; //!< b = (2^0.5) * cos(    pi /  8);  Used in forward and inverse DCT.
//float C_c = 1.175875602419359f; //!< c = (2^0.5) * cos(3 * pi / 16);  Used in forward and inverse DCT.
//float C_d = 0.785694958387102f; //!< d = (2^0.5) * cos(5 * pi / 16);  Used in forward and inverse DCT.
//float C_e = 0.541196100146197f; //!< e = (2^0.5) * cos(3 * pi /  8);  Used in forward and inverse DCT.
//float C_f = 0.275899379282943f; //!< f = (2^0.5) * cos(7 * pi / 16);  Used in forward and inverse DCT.
//float C_norm = 0.3535533905932737f; // 1 / (8^0.5)

/*Already implemented in utils/BmpUtil.cpp*/
//float round_f(float num)
//{
//    float NumAbs = fabs(num);
//    int NumAbsI = (int)(NumAbs + 0.5f);
//    float sign = num > 0 ? 1.0f : -1.0f;
//    return sign * NumAbsI;
//}

/*
    GPU dct8x8
    Reference: samples/3_Imaging/dct8x8/dct8x8.cu: CUDA2
*/
void GpuKernel::dct8x8_2d(KernelParams& kernel_params, void** in_img, void** out_img){
    /* integration code */
    float* ImgF1   = reinterpret_cast<float*>(*in_img);
    float* out_tmp = reinterpret_cast<float*>(*out_img);
    // a hard-coded params that used by this kernel.
    //int ImgStride;

    //allocate device memory
    float *src, *dst;
    ROI Size;
    Size.width  = kernel_params.params.get_kernel_size();
    Size.height = kernel_params.params.get_kernel_size();
    /* integration code */
    int StrideF = (((int)ceil((Size.width*sizeof(float))/16.0f))*16) / sizeof(float);
 //   byte *ImgDst = MallocPlaneByte(Size.width, Size.height, &ImgStride);
    size_t DeviceStride;
    checkCudaErrors(cudaMallocPitch((void **)&src, &DeviceStride, Size.width * sizeof(float), Size.height));
    checkCudaErrors(cudaMallocPitch((void **)&dst, &DeviceStride, Size.width * sizeof(float), Size.height));
    DeviceStride /= sizeof(float);

    //copy from host memory to device
    checkCudaErrors(cudaMemcpy2D(src, DeviceStride * sizeof(float),
                                 ImgF1, StrideF * sizeof(float),
                                 Size.width * sizeof(float), Size.height,
                                 cudaMemcpyHostToDevice));

    dim3 GridFullWarps(Size.width / KER2_BLOCK_WIDTH, Size.height / KER2_BLOCK_HEIGHT, 1);
    dim3 ThreadsFullWarps(8, KER2_BLOCK_WIDTH/8, KER2_BLOCK_HEIGHT/8);

    //perform block-wise DCT processing and benchmarking
    const int numIterations = 100;

    for (int i = -1; i < numIterations; i++)
    {
        if (i == 0)
        {
            checkCudaErrors(cudaDeviceSynchronize());
        }
 
        CUDAkernel2DCT<<<GridFullWarps, ThreadsFullWarps>>>(dst, src, (int)DeviceStride);
        getLastCudaError("Kernel execution failed");
    }
 
    checkCudaErrors(cudaDeviceSynchronize());

    //setup execution parameters for quantization
    dim3 ThreadsSmallBlocks(BLOCK_SIZE, BLOCK_SIZE);
    dim3 GridSmallBlocks(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

    // execute Quantization kernel
    CUDAkernelQuantizationFloat<<< GridSmallBlocks, ThreadsSmallBlocks >>>(dst, (int) DeviceStride);
    getLastCudaError("Kernel execution failed");
 
    //perform block-wise IDCT processing
    CUDAkernel2IDCT<<<GridFullWarps, ThreadsFullWarps >>>(src, dst, (int)DeviceStride);
    checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("Kernel execution failed");
    
    //copy quantized image block to host
    checkCudaErrors(cudaMemcpy2D(out_tmp, StrideF *sizeof(float),
                                 src, DeviceStride *sizeof(float),
                                 Size.width *sizeof(float), Size.height,
                                 cudaMemcpyDeviceToHost));
 
    //convert image back to byte representation
//    AddFloatPlane(128.0f, out_tmp, StrideF, Size);
//    CopyFloat2Byte(out_tmp, StrideF, ImgDst, ImgStride, Size);

    //clean up memory
    checkCudaErrors(cudaFree(dst));
    checkCudaErrors(cudaFree(src));
}


#include <string>
#include <stdio.h>
#include "cuda_utils.h"
#include "kernels_gpu.h"
#include "kernels_fft.cuh"
#include "kernels_fft_wrapper.cu"

#include <cuda_runtime.h>
#include <cufft.h>
//#include <cuda_runtime_api.h>
//#include <cuda.h>


void GpuKernel::fft_2d_input_conversion(){
    this->input_array_type.device_fp  = this->input_array_type.host_fp;
}

void GpuKernel::fft_2d_output_conversion(){
    Mat result;
    const int kernelH = 7;
    const int kernelW = 6;
    const int   dataH = kernel_params.params.get_kernel_size();
    const int   dataW = kernel_params.params.get_kernel_size();
    const int    fftH = snapTransformSize(dataH + kernelH - 1); 
    const int    fftW = snapTransformSize(dataW + kernelW - 1); 

    assert(this->output_array_type.device_fp != NULL);

    array2mat(result, this->output_array_type.device_fp, fftH, fftW);
    Mat cropped = result(Range(0, dataH), Range(0, dataW)); 
    mat2array(cropped, this->output_array_type.host_fp);
}    

/*
    GPU convolveFFT2D, this kernel used a fixed 7x6 convolving kernel.
    Reference: samples/3_Imaging/convolutionFFT2D/convolutionFFT2D.cu
*/
void GpuKernel::fft_2d(KernelParams& kernel_params, void** in_array, void** out_array){
    float* h_Data      = reinterpret_cast<float*>(*in_array);
    float* h_ResultGPU = reinterpret_cast<float*>(*out_array);
    
    float* h_Kernel;
    
    float* d_Data ;
    float* d_PaddedData;
    float* d_Kernel;
    float* d_PaddedKernel;

    fComplex* d_DataSpectrum;
    fComplex* d_KernelSpectrum;

    cufftHandle fftPlanFwd, fftPlanInv;

    const int kernelH = 7;
    const int kernelW = 6;
    const int kernelY = 3;
    const int kernelX = 4;
    const int   dataH = kernel_params.params.get_kernel_size();
    const int   dataW = kernel_params.params.get_kernel_size();
    const int    fftH = snapTransformSize(dataH + kernelH - 1); 
    const int    fftW = snapTransformSize(dataW + kernelW - 1); 

    //printf("...allocating memory\n");
    float fft_2d_kernel_array[7*6] = {
        13, 12, 13,  0,  1,  1,
        0,  7,  8,  2,  8,  0,
        5,  9,  1, 11, 11,  3,
        14, 14,  8, 11,  0,  3,
        6,  8, 14, 13,  0, 10,
        10, 11, 14,  1,  2,  0,
        5, 15,  7,  5,  1,  7
    };
    h_Kernel = fft_2d_kernel_array;

    checkCudaErrors(cudaMalloc((void **)&d_Data, dataH * dataW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));
    
    checkCudaErrors(cudaMalloc((void **)&d_PaddedData,   fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum,   fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    checkCudaErrors(cudaMemset(d_KernelSpectrum, 0, fftH * (fftW / 2 + 1) * sizeof(fComplex)));

    //printf("...creating R2C & C2R FFT plans for %i x %i\n", fftH, fftW);
    checkCudaErrors(cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C));
    checkCudaErrors(cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R));

    //printf("...uploading to gpu and padding convolution kernel and input data\n");
    checkCudaErrors(cudaMemcpy(d_Kernel, 
                               h_Kernel, 
                               kernelH * kernelW * sizeof(float), 
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Data,   
                               h_Data,   
                               dataH   * dataW *   sizeof(float), 
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMemset(d_PaddedData,   0, fftH * fftW * sizeof(float)));

    padKernel(
        d_PaddedKernel,
        d_Kernel,
        fftH,
        fftW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );

    padDataClampToBorder(
        d_PaddedData,
        d_Data,
        fftH,
        fftW,
        dataH,
        dataW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );

    //printf("...transforming convolution kernel\n");
    checkCudaErrors(cufftExecR2C(fftPlanFwd, 
                 (cufftReal *)d_PaddedKernel, 
                 (cufftComplex *)d_KernelSpectrum));
    
    //printf("...running GPU FFT convolution\n");
    checkCudaErrors(cudaDeviceSynchronize());
    
    checkCudaErrors(cufftExecR2C(fftPlanFwd, 
                                 (cufftReal *)d_PaddedData, 
                                 (cufftComplex *)d_DataSpectrum));
    modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
    checkCudaErrors(cufftExecC2R(fftPlanInv, 
                                 (cufftComplex *)d_DataSpectrum, 
                                 (cufftReal *)d_PaddedData));
 
    checkCudaErrors(cudaDeviceSynchronize());
    
    float* tmp = (float *)malloc(fftH    * fftW * sizeof(float));;
    
    //printf("...reading back GPU convolution results\n");
    checkCudaErrors(cudaMemcpy(tmp, d_PaddedData, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost)); 
    h_ResultGPU = tmp;
    *out_array = (void*)h_ResultGPU;

    checkCudaErrors(cufftDestroy(fftPlanInv));
    checkCudaErrors(cufftDestroy(fftPlanFwd));

    checkCudaErrors(cudaFree(d_DataSpectrum));
    checkCudaErrors(cudaFree(d_KernelSpectrum));
    checkCudaErrors(cudaFree(d_PaddedData));
    checkCudaErrors(cudaFree(d_PaddedKernel));
    checkCudaErrors(cudaFree(d_Data));
    checkCudaErrors(cudaFree(d_Kernel));
}

#include <assert.h>
#include <iostream>
#include "cuda_utils.h"
#include "kernels_fft.cuh"

////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
//extern "C" void padKernel(
//    float *d_Dst,
//    float *d_Src,
//    int fftH,
//    int fftW,
//    int kernelH,
//    int kernelW,
//    int kernelY,
//    int kernelX
//)
//{
//    assert(d_Src != d_Dst);
//    dim3 threads(32, 8);
//    dim3 grid(iDivUp(kernelW, threads.x), iDivUp(kernelH, threads.y));
// 
//    SET_FLOAT_BASE;
//#if (USE_TEXTURE)
//    cudaTextureObject_t texFloat;
//    cudaResourceDesc    texRes;
//    memset(&texRes,0,sizeof(cudaResourceDesc));
// 
//    texRes.resType            = cudaResourceTypeLinear;
//    texRes.res.linear.devPtr    = d_Src;
//    texRes.res.linear.sizeInBytes = sizeof(float)*kernelH*kernelW;
//    texRes.res.linear.desc = cudaCreateChannelDesc<float>();
//
//    cudaTextureDesc             texDescr;
//    memset(&texDescr,0,sizeof(cudaTextureDesc));
//    texDescr.normalizedCoords = false;
//    texDescr.filterMode       = cudaFilterModeLinear;
//    texDescr.addressMode[0] = cudaAddressModeWrap;
//    texDescr.readMode = cudaReadModeElementType;
//  
//    cudaCreateTextureObject(&texFloat, &texRes, &texDescr, NULL);
//#endif
//  
//    padKernel_kernel<<<grid, threads>>>(
//        d_Dst,
//        d_Src,
//        fftH,
//        fftW,
//        kernelH,
//        kernelW,
//        kernelY,
//        kernelX
//#if (USE_TEXTURE)
//        , texFloat
//#endif
//    );
////    getLastCudaError("padKernel_kernel<<<>>> execution failed\n");
// 
//#if (USE_TEXTURE)
//    cudaDestroyTextureObject(texFloat);
//#endif
//}

////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
//extern "C" void padDataClampToBorder(
//    float *d_Dst,
//    float *d_Src,
//    int fftH,
//    int fftW,
//    int dataH,
//    int dataW,
//    int kernelW,
//    int kernelH,
//    int kernelY,
//    int kernelX
//)
//{
//    assert(d_Src != d_Dst);
//    dim3 threads(32, 8);
//    dim3 grid(iDivUp(fftW, threads.x), iDivUp(fftH, threads.y));
//
//#if (USE_TEXTURE)
//    cudaTextureObject_t texFloat;
//    cudaResourceDesc            texRes;
//    memset(&texRes,0,sizeof(cudaResourceDesc));
//
//    texRes.resType            = cudaResourceTypeLinear;
//    texRes.res.linear.devPtr    = d_Src;
//    texRes.res.linear.sizeInBytes = sizeof(float)*dataH*dataW;
//    texRes.res.linear.desc = cudaCreateChannelDesc<float>();
// 
//    cudaTextureDesc             texDescr;
//    memset(&texDescr,0,sizeof(cudaTextureDesc));
// 
//    texDescr.normalizedCoords = false;
//    texDescr.filterMode       = cudaFilterModeLinear;
//    texDescr.addressMode[0] = cudaAddressModeWrap;
//    texDescr.readMode = cudaReadModeElementType;
// 
//    cudaCreateTextureObject(&texFloat, &texRes, &texDescr, NULL);
//#endif
// 
//    padDataClampToBorder_kernel<<<grid, threads>>>(
//        d_Dst,
//        d_Src,
//        fftH,
//        fftW,
//        dataH,
//        dataW,
//        kernelH,
//        kernelW,
//        kernelY,
//        kernelX
//#if (USE_TEXTURE)
//       ,texFloat
//#endif
//    );
////    getLastCudaError("padDataClampToBorder_kernel<<<>>> execution fai    led\n");
// 
//#if (USE_TEXTURE)
//    cudaDestroyTextureObject(texFloat);
//#endif
//}

////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
//extern "C" void modulateAndNormalize(
//    fComplex *d_Dst,
//    fComplex *d_Src,
//    int fftH,
//    int fftW,
//    int padding
//)
//{
//    assert(fftW % 2 == 0);
//    const int dataSize = fftH * (fftW / 2 + padding);
//
//    modulateAndNormalize_kernel<<<iDivUp(dataSize, 256), 256>>>(
//        d_Dst,
//        d_Src,
//        dataSize,
//        1.0f / (float)(fftW *fftH)
//    );
////    getLastCudaError("modulateAndNormalize() execution failed\n");
//}


//void fft_2d_input_conversion_wrapper(){
//    return;
//}

#include "kernels_gpu.h"

#include <cuda_runtime.h>

#ifdef RD_WG_SIZE_0_0                                                            
        #define BLOCK_SIZE RD_WG_SIZE_0_0                                        
#elif defined(RD_WG_SIZE_0)                                                      
        #define BLOCK_SIZE RD_WG_SIZE_0                                          
#elif defined(RD_WG_SIZE)                                                        
        #define BLOCK_SIZE RD_WG_SIZE                                            
#else
        #define BLOCK_SIZE 16                                                            
#endif

/* some constants */
#define chip_height 0.016
#define chip_width 0.016
#define t_chip 0.0005
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
#define FACTOR_CHIP 0.5
#define MAX_PD 3.0e6

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
//#define MIN(a, b) ((a)<=(b) ? (a) : (b))
 
__global__ void calculate_temp(int iteration,  //number of iteration
                               float *power,   //power input
                               float *temp_src,    //temperature input/output
                               float *temp_dst,    //temperature input/output
                               int grid_cols,  //Col of grid
                               int grid_rows,  //Row of grid
                               int border_cols,  // border offset 
                               int border_rows,  // border offset
                               float Cap,      //Capacitance
                               float Rx,
                               float Ry,
                               float Rz,
                               float step,
                               float time_elapsed){

        __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result
 
    float amb_temp = 80.0;
        float step_div_Cap;
        float Rx_1,Ry_1,Rz_1;

    int bx = blockIdx.x;
        int by = blockIdx.y;
 
    int tx=threadIdx.x;
    int ty=threadIdx.y;
 
    step_div_Cap=step/Cap;

    Rx_1=1/Rx;
    Ry_1=1/Ry;
    Rz_1=1/Rz;
 
        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data
 
        // calculate the small block size
    int small_block_rows = BLOCK_SIZE-iteration*2;//EXPAND_RATE
    int small_block_cols = BLOCK_SIZE-iteration*2;//EXPAND_RATE

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkY = small_block_rows*by-border_rows;
        int blkX = small_block_cols*bx-border_cols;
        int blkYmax = blkY+BLOCK_SIZE-1;
        int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
    int yidx = blkY+ty;
    int xidx = blkX+tx;
 
        // load data if it is within the valid input range
    int loadYidx=yidx, loadXidx=xidx;
        int index = grid_cols*loadYidx+loadXidx;
 
    if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
            temp_on_cuda[ty][tx] = temp_src[index];  // Load the temperature data from global memory to shared memory
            power_on_cuda[ty][tx] = power[index];// Load the power data from global memory to shared memory
    }
    __syncthreads();

        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validYmin = (blkY < 0) ? -blkY : 0;
        int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE-1-(blkYmax-grid_rows+1) : BLOCK_SIZE-1;
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE-1-(blkXmax-grid_cols+1) : BLOCK_SIZE-1;
 
        int N = ty-1;
        int S = ty+1;
        int W = tx-1;
        int E = tx+1;
 
        N = (N < validYmin) ? validYmin : N;
        S = (S > validYmax) ? validYmax : S;
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;
 
        bool computed;
        for (int i=0; i<iteration ; i++){
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(tx, validXmin, validXmax) && \
                  IN_RANGE(ty, validYmin, validYmax) ) {
                  computed = true;
                  temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] +
                     (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0*temp_on_cuda[ty][tx]) * Ry_1 +
                     (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0*temp_on_cuda[ty][tx]) * Rx_1 +
                     (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
 
            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)     //Assign the computation range
                temp_on_cuda[ty][tx]= temp_t[ty][tx];
            __syncthreads();
          }
 
      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if (computed){
          temp_dst[index]= temp_t[ty][tx];
      }
}
/*
   compute N time steps
*/
 
int compute_tran_temp(float *MatrixPower,float *MatrixTemp[2], int col, int row, \
        int total_iterations, int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows)
{
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(blockCols, blockRows);
     
    float grid_height = chip_height / row;
    float grid_width = chip_width / col;
     
    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);
     
    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope;
    float t;
        float time_elapsed;
    time_elapsed=0.001;
 
        int src = 1, dst = 0;
 
    for (t = 0; t < total_iterations; t+=num_iterations) {
            int temp = src;
            src = dst;
            dst = temp;
            calculate_temp<<<dimGrid, dimBlock>>>(MIN(num_iterations, total_iterations-t), MatrixPower,MatrixTemp[src],MatrixTemp[dst],\
        col,row,borderCols, borderRows, Cap,Rx,Ry,Rz,step,time_elapsed);
    }
        return dst;
}

/* Reference code: rodinia_3.1/cuda/hotspot/hotspot.cu */
void GpuKernel::hotspot_2d(KernelParams& kernel_params, void** input, void** output){

    int dim = kernel_params.params.get_kernel_size();
    int grid_rows = dim;
    int grid_cols = dim;
    int size = dim * dim;

    /* some constants */
    int total_iterations = 1;
    int pyramid_height = 1;

    /* pyramid parameters */
    # define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline
    int borderCols = (pyramid_height)*EXPAND_RATE/2;
    int borderRows = (pyramid_height)*EXPAND_RATE/2;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

    /* host pointers */
    float* host_float_ptr = reinterpret_cast<float*>(*input);
    float* FilesavingTemp = host_float_ptr;
    float* FilesavingPower = &host_float_ptr[size];
    float* MatrixOut = reinterpret_cast<float*>(*output);

    /* device pointers */
    float *MatrixTemp[2], *MatrixPower;
    cudaMalloc((void**)&MatrixTemp[0], sizeof(float)*size);
    cudaMalloc((void**)&MatrixTemp[1], sizeof(float)*size);
    cudaMemcpy(MatrixTemp[0], FilesavingTemp, sizeof(float)*size, cudaMemcpyHostToDevice);
 
    cudaMalloc((void**)&MatrixPower, sizeof(float)*size);
    cudaMemcpy(MatrixPower, FilesavingPower, sizeof(float)*size, cudaMemcpyHostToDevice);
    printf("Start computing the transient temperature\n");
    int ret = compute_tran_temp(MatrixPower,MatrixTemp,grid_cols,grid_rows, \
     total_iterations,pyramid_height, blockCols, blockRows, borderCols, borderRows);
    printf("Ending simulation\n");
    cudaMemcpy(MatrixOut, MatrixTemp[ret], sizeof(float)*size, cudaMemcpyDeviceToHost);
 
    //writeoutput(MatrixOut,grid_rows, grid_cols, ofile);
 
    cudaFree(MatrixPower);
    cudaFree(MatrixTemp[0]);
    cudaFree(MatrixTemp[1]);
}
#include "kernels_gpu.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

void GpuKernel::kmeans_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){
}
#include <string>
#include <stdio.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include "kernels_gpu.h"

void GpuKernel::laplacian_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){
    auto laplacian = cuda::createLaplacianFilter(in_img.type(), in_img.type(), 3/*kernel size*/, 1/*scale*/, BORDER_DEFAULT);
    laplacian->apply(in_img, out_img);
    cuda::abs(out_img, out_img);
}
#include <string>
#include <stdio.h>
#include <opencv2/cudafilters.hpp> // create[XXX]Filter()
#include "kernels_gpu.h"

void GpuKernel::mean_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){
    auto median = cuda::createBoxFilter(in_img.type(), in_img.type(), Size(3, 3),     Point(-1, -1), BORDER_DEFAULT);
    median->apply(in_img, out_img);
}
#include <string>
#include <stdio.h>
#include "kernels_gpu.h"

void GpuKernel::minimum_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){
    out_img = in_img;
}
#include <string>
#include <stdio.h>
#include <opencv2/cudaarithm.hpp> // addWeighted()
#include <opencv2/cudafilters.hpp> // create[XXX]Filter()
#include "kernels_gpu.h"
void GpuKernel::sobel_2d(const cuda::GpuMat in_img, cuda::GpuMat& out_img){

    cuda::GpuMat grad_x, grad_y;
    cuda::GpuMat abs_grad_x, abs_grad_y;

    int ddepth = CV_32F;
    auto sobel_dx = cuda::createSobelFilter(in_img.type(), ddepth, 1, 0, 3);
    auto sobel_dy = cuda::createSobelFilter(in_img.type(), ddepth, 0, 1, 3);
 
    sobel_dx->apply(in_img, grad_x);
    sobel_dy->apply(in_img, grad_y);
 
    cuda::abs(grad_x, abs_grad_x);
    cuda::abs(grad_y, abs_grad_y);
  
    cuda::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, out_img);
}
#include "srad.h"
#include "kernels_gpu.h"
#include "srad_kernel.cu"


void GpuKernel::srad_2d(KernelParams& kernel_params, void** input, void** output){
    int rows = kernel_params.params.get_kernel_size();
    int cols = kernel_params.params.get_kernel_size();
    int size_I, size_R, niter = 1, iter;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI, varROI;

    float *J_cuda;
    float *C_cuda;
    float *E_C, *W_C, *N_C, *S_C;

    unsigned int r1 = 0, r2 = rows-1, c1 = 0, c2 = cols-1; // need init
    float *c;

    size_I = cols * rows;
    size_R = (r1-r1+1)*(c2-c1+1);
 
    I = (float*)*input;
    J = (float*)*output;
    c = (float *)malloc(sizeof(float)* size_I);

    //Allocate device memory
    cudaMalloc((void**)& J_cuda, sizeof(float)* size_I);
    cudaMalloc((void**)& C_cuda, sizeof(float)* size_I);
    cudaMalloc((void**)& E_C, sizeof(float)* size_I);
    cudaMalloc((void**)& W_C, sizeof(float)* size_I);
    cudaMalloc((void**)& S_C, sizeof(float)* size_I);
    cudaMalloc((void**)& N_C, sizeof(float)* size_I);

    for (int k = 0;  k < size_I; k++ ) {
        J[k] = (float)exp(I[k]) ;
    }

    for(iter=0; iter < niter ; iter++){
        sum=0; sum2=0;
        for (int i=r1; i<=r2; i++) {
            for (int j=c1; j<=c2; j++) {
                tmp   = J[i * cols + j];
                sum  += tmp ;
                sum2 += tmp*tmp;
            }
        }
        meanROI = sum / size_R;
        varROI  = (sum2 / size_R) - meanROI*meanROI;
        q0sqr   = varROI / (meanROI*meanROI);

        //Currently the input size must be divided by 16 - the block size
        int block_x = cols/BLOCK_SIZE ;
        int block_y = rows/BLOCK_SIZE ;
 
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(block_x , block_y);
 
        //Copy data from main memory to device memory
        cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice);
 
        //Run kernels
        srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda,     cols, rows, q0sqr);
        srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda,     cols, rows, lambda, q0sqr);
 
        //Copy data from device memory to main memory
        cudaMemcpy(J, J_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    //cudaThreadSynchronize();

    cudaFree(C_cuda);
    cudaFree(J_cuda);
    cudaFree(E_C);
    cudaFree(W_C);
    cudaFree(N_C);
    cudaFree(S_C);
    free(c);
}
#include "srad.h"
#include <stdio.h>

//__global__ void
//srad_cuda_1(
//		  float *E_C, 
//		  float *W_C, 
//		  float *N_C, 
//		  float *S_C,
//		  float * J_cuda, 
//		  float * C_cuda, 
//		  int cols, 
//		  int rows, 
//		  float q0sqr
//) 
//{
//
//  //block id
//  int bx = blockIdx.x;
//  int by = blockIdx.y;
//
//  //thread id
//  int tx = threadIdx.x;
//  int ty = threadIdx.y;
//  
//  //indices
//  int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
//  int index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
//  int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
//  int index_w = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
//  int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
//
//  float n, w, e, s, jc, g2, l, num, den, qsqr, c;
//
//  //shared memory allocation
//  __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
//  __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE];
//
//  __shared__ float north[BLOCK_SIZE][BLOCK_SIZE];
//  __shared__ float south[BLOCK_SIZE][BLOCK_SIZE];
//  __shared__ float  east[BLOCK_SIZE][BLOCK_SIZE];
//  __shared__ float  west[BLOCK_SIZE][BLOCK_SIZE];
//
//  //load data to shared memory
//  north[ty][tx] = J_cuda[index_n]; 
//  south[ty][tx] = J_cuda[index_s];
//  if ( by == 0 ){
//  north[ty][tx] = J_cuda[BLOCK_SIZE * bx + tx]; 
//  }
//  else if ( by == gridDim.y - 1 ){
//  south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx];
//  }
//   __syncthreads();
// 
//  west[ty][tx] = J_cuda[index_w];
//  east[ty][tx] = J_cuda[index_e];
//
//  if ( bx == 0 ){
//  west[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + cols * ty]; 
//  }
//  else if ( bx == gridDim.x - 1 ){
//  east[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1];
//  }
// 
//  __syncthreads();
//  
// 
//
//  temp[ty][tx]      = J_cuda[index];
//
//  __syncthreads();
//
//   jc = temp[ty][tx];
//
//   if ( ty == 0 && tx == 0 ){ //nw
//	n  = north[ty][tx] - jc;
//    s  = temp[ty+1][tx] - jc;
//    w  = west[ty][tx]  - jc; 
//    e  = temp[ty][tx+1] - jc;
//   }	    
//   else if ( ty == 0 && tx == BLOCK_SIZE-1 ){ //ne
//	n  = north[ty][tx] - jc;
//    s  = temp[ty+1][tx] - jc;
//    w  = temp[ty][tx-1] - jc; 
//    e  = east[ty][tx] - jc;
//   }
//   else if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
//	n  = temp[ty-1][tx] - jc;
//    s  = south[ty][tx] - jc;
//    w  = temp[ty][tx-1] - jc; 
//    e  = east[ty][tx]  - jc;
//   }
//   else if ( ty == BLOCK_SIZE -1 && tx == 0 ){//sw
//	n  = temp[ty-1][tx] - jc;
//    s  = south[ty][tx] - jc;
//    w  = west[ty][tx]  - jc; 
//    e  = temp[ty][tx+1] - jc;
//   }
//
//   else if ( ty == 0 ){ //n
//	n  = north[ty][tx] - jc;
//    s  = temp[ty+1][tx] - jc;
//    w  = temp[ty][tx-1] - jc; 
//    e  = temp[ty][tx+1] - jc;
//   }
//   else if ( tx == BLOCK_SIZE -1 ){ //e
//	n  = temp[ty-1][tx] - jc;
//    s  = temp[ty+1][tx] - jc;
//    w  = temp[ty][tx-1] - jc; 
//    e  = east[ty][tx] - jc;
//   }
//   else if ( ty == BLOCK_SIZE -1){ //s
//	n  = temp[ty-1][tx] - jc;
//    s  = south[ty][tx] - jc;
//    w  = temp[ty][tx-1] - jc; 
//    e  = temp[ty][tx+1] - jc;
//   }
//   else if ( tx == 0 ){ //w
//	n  = temp[ty-1][tx] - jc;
//    s  = temp[ty+1][tx] - jc;
//    w  = west[ty][tx] - jc; 
//    e  = temp[ty][tx+1] - jc;
//   }
//   else{  //the data elements which are not on the borders 
//	n  = temp[ty-1][tx] - jc;
//    s  = temp[ty+1][tx] - jc;
//    w  = temp[ty][tx-1] - jc; 
//    e  = temp[ty][tx+1] - jc;
//   }
//
//
//    g2 = ( n * n + s * s + w * w + e * e ) / (jc * jc);
//
//    l = ( n + s + w + e ) / jc;
//
//	num  = (0.5*g2) - ((1.0/16.0)*(l*l)) ;
//	den  = 1 + (.25*l);
//	qsqr = num/(den*den);
//
//	// diffusion coefficent (equ 33)
//	den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
//	c = 1.0 / (1.0+den) ;
//
//    // saturate diffusion coefficent
//	if (c < 0){temp_result[ty][tx] = 0;}
//	else if (c > 1) {temp_result[ty][tx] = 1;}
//	else {temp_result[ty][tx] = c;}
//
//    __syncthreads();
//
//    C_cuda[index] = temp_result[ty][tx];
//	E_C[index] = e;
//	W_C[index] = w;
//	S_C[index] = s;
//	N_C[index] = n;
//
//}
//
//__global__ void
//srad_cuda_2(
//		  float *E_C, 
//		  float *W_C, 
//		  float *N_C, 
//		  float *S_C,	
//		  float * J_cuda, 
//		  float * C_cuda, 
//		  int cols, 
//		  int rows, 
//		  float lambda,
//		  float q0sqr
//) 
//{
//	//block id
//	int bx = blockIdx.x;
//    int by = blockIdx.y;
//
//	//thread id
//    int tx = threadIdx.x;
//    int ty = threadIdx.y;
//
//	//indices
//    int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
//	int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
//    int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
//	float cc, cn, cs, ce, cw, d_sum;
//
//	//shared memory allocation
//	__shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE];
//    __shared__ float  east_c[BLOCK_SIZE][BLOCK_SIZE];
//
//    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE];
//    __shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE];
//    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
//
//    //load data to shared memory
//	temp[ty][tx]      = J_cuda[index];
//
//    __syncthreads();
//	 
//	south_c[ty][tx] = C_cuda[index_s];
//
//	if ( by == gridDim.y - 1 ){
//	south_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx];
//	}
//	__syncthreads();
//	 
//	 
//	east_c[ty][tx] = C_cuda[index_e];
//	
//	if ( bx == gridDim.x - 1 ){
//	east_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1];
//	}
//	 
//    __syncthreads();
//  
//    c_cuda_temp[ty][tx]      = C_cuda[index];
//
//    __syncthreads();
//
//	cc = c_cuda_temp[ty][tx];
//
//   if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
//	cn  = cc;
//    cs  = south_c[ty][tx];
//    cw  = cc; 
//    ce  = east_c[ty][tx];
//   } 
//   else if ( tx == BLOCK_SIZE -1 ){ //e
//	cn  = cc;
//    cs  = c_cuda_temp[ty+1][tx];
//    cw  = cc; 
//    ce  = east_c[ty][tx];
//   }
//   else if ( ty == BLOCK_SIZE -1){ //s
//	cn  = cc;
//    cs  = south_c[ty][tx];
//    cw  = cc; 
//    ce  = c_cuda_temp[ty][tx+1];
//   }
//   else{ //the data elements which are not on the borders 
//	cn  = cc;
//    cs  = c_cuda_temp[ty+1][tx];
//    cw  = cc; 
//    ce  = c_cuda_temp[ty][tx+1];
//   }
//
//   // divergence (equ 58)
//   d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];
//
//   // image update (equ 61)
//   c_cuda_result[ty][tx] = temp[ty][tx] + 0.25 * lambda * d_sum;
//
//   __syncthreads();
//              
//   J_cuda[index] = c_cuda_result[ty][tx];
//    
//}
