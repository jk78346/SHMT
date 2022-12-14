#include <iostream>
#include "kernels_fft.cuh"

__global__ void test_kernel(){
  printf("Hello World!\n");
}

void fft_2d_kernel_wrapper(float* in_img, float* out_img) {
  std::cout << __func__ << ": calling a testing cuda kernel function..." << std::endl;
  test_kernel<<<1, 1>>>();
  return;
}
