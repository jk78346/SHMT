#ifndef __KERNELS_H__
#define __KERNELS_H__
#include <opencv2/opencv.hpp>

using namespace cv;

void sobel_2d_cpu(Mat& in_img, Mat& out_img);
typedef void (*func_ptr)(Mat&, Mat&);

#endif
