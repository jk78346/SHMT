#ifndef __UTILS_H__
#define __UTILS_H__
#include <chrono>
#include <string.h>
#include <opencv2/opencv.hpp>
#include "types.h"

using namespace cv;

double get_time_ms(timing end, timing start);
void read_img(const std::string file_name, int rows, int cols, Mat& img);
void mat2array(Mat img, float* data);
//void mat2array_v2(Mat img, float* data);
void array2mat(Mat& img, float* data, int CV_type, int rows, int cols);
std::string get_edgetpu_kernel_path(std::string app_name, int shape0, int shape1);

#endif
