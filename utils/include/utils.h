#ifndef __UTILS_H__
#define __UTILS_H__
#include <chrono>
#include <opencv2/opencv.hpp>
#include "types.h"

using namespace cv;

double get_time_ms(timing end, timing start);
void read_img(const std::string file_name, int rows, int cols, Mat& img);
void mat2array(Mat& img, float* data);
void array2mat(Mat& img, float* data, int CV_type, int rows, int cols);

#endif
