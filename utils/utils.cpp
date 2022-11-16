#include <fstream>
#include <iostream>
#include <assert.h>
#include "utils.h"

double get_time_ms(timing end, timing start){
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/1000000.0;
}

/*
    Read a image in file into opencv Mat. 
*/
void read_img(const std::string file_name, int rows, int cols, Mat& img){
    Mat raw = imread(file_name);
    assert(!raw.empty());
    cvtColor(raw, img, COLOR_BGR2GRAY);
    resize(img, img, Size(rows, cols), 0, 0, INTER_AREA);
    assert(img.size().width * img.size().height == rows * cols);
}

void mat2array(Mat& img, float* data){
    // data has to be pre-allocated with proper size
    if(!img.isContinuous()){
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

void array2mat(Mat& img, float* data, int CV_type, int rows, int cols){
    Mat tmp = Mat(rows, cols, CV_type, data);
    tmp.copyTo(img);
}

std::string get_edgetpu_kernel_path(std::string app_name, int shape0, int shape1){
    std::string path =  "../models/"+ 
                        app_name+"_"+std::to_string(shape0)+"x"+std::to_string(shape1)+"/"+
                        app_name+"_edgetpu.tflite";
    std::ifstream ifile(path);
    assert(ifile);
    return path;
}

