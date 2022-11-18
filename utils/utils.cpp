#include <fstream>
#include <iostream>
#include <assert.h>
#include "utils.h"

/* Mat type info:
    https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
*/

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

void mat2array(Mat img, float* data){
    // data has to be pre-allocated with proper size
    if(!img.isContinuous()){
        img = img.clone();
    }
    // row-major
    if(img.type() % 8 <= 1){ // CV_8U, CV_8S series
        for(int i = 0 ; i < img.rows ; i++){
            for(int j = 0 ; j < img.cols ; j++){
                int idx = i*(img.cols)+j;
                data[idx] = img.data[idx]; // uint8_t to float conversion
            }
        }
    }else if(img.type() % 8 == 5){ // CV_32F series
        std::memcpy(data, 
                    (float*)img.data, 
                    img.size().width * img.size().height * sizeof(float));
    }else{
        std::cout << "[WARN] " << __func__ << ": undefined img.type: " 
                  << img.type() << " to float* conversion." << std::endl;
    }
}

void array2mat(Mat& img, float* data, int CV_type, int rows, int cols){
    if(CV_type % 8 == 0 || CV_type % 8 == 1 || CV_type % 8 == 5){
        Mat tmp = Mat(rows, cols, CV_type, data);
        tmp.copyTo(img);
    }else{
        std::cout << "[WARN] " << __func__ << ": undefined float* to img.type: " 
                  << img.type() << " conversion." << std::endl;
    }
}

std::string get_edgetpu_kernel_path(std::string app_name, int shape0, int shape1){
    std::string path =  "../models/"+ 
                        app_name+"_"+std::to_string(shape0)+"x"+std::to_string(shape1)+"/"+
                        app_name+"_edgetpu.tflite";
    std::ifstream ifile(path);
    assert(ifile);
    return path;
}

