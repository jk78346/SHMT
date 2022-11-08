#include <string>
#include <stdio.h>
#include <kernels.h>
#include "utils.h"
#include <unordered_map>
#include <opencv2/cudafilters.hpp> // create[XXX]Filter()
#include <opencv2/cudaarithm.hpp> // addWeighted()

std::unordered_map<std::string, func_ptr_gpu> gpu_func_table = {
    {"minimum_2d", minimum_2d_gpu},
    {"sobel_2d", sobel_2d_gpu},
    {"mean_2d", mean_2d_gpu},
    {"laplacian_2d", laplacian_2d_gpu}
};

void minimum_2d_gpu(cuda::GpuMat& in_img, cuda::GpuMat& out_img){
    out_img = in_img;
}

void sobel_2d_gpu(cuda::GpuMat& in_img, cuda::GpuMat& out_img){

    cuda::GpuMat grad_x, grad_y;
    cuda::GpuMat abs_grad_x, abs_grad_y;

    auto sobel_dx = cuda::createSobelFilter(in_img.type(), in_img.type(), 1, 0, 3);
    auto sobel_dy = cuda::createSobelFilter(in_img.type(), in_img.type(), 0, 1, 3);

    sobel_dx->apply(in_img, grad_x);
    sobel_dy->apply(in_img, grad_y);
    
    cuda::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, out_img);
}

void mean_2d_gpu(cuda::GpuMat& in_img, cuda::GpuMat& out_img){
    auto median = cuda::createBoxFilter(in_img.type(), in_img.type(), Size(3, 3), Point(-1, -1), BORDER_DEFAULT);
    median->apply(in_img, out_img);
}

void laplacian_2d_gpu(cuda::GpuMat& in_img, cuda::GpuMat& out_img){
    int ddepth = CV_32F;
//    Laplacian(in_img, out_img, ddepth, 3/*kernel size*/, 1/*scale*/, 0/*delta*/, BORDER_DEFAULT);
//    convertScaleAbs(out_img, out_img);
    auto lap = cuda::createLaplacianFilter(in_img.type(), in_img.type(), 3/*kernel size*/, 1/*scale*/, BORDER_DEFAULT);
    lap->apply(in_img, out_img);
}




