#include <fstream>
#include <iostream>
#include <assert.h>
#include "utils.h"

/* Mat type info:
    https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
*/

#define ASSERT_WITH_MESSAGE(condition, message) do { \
if (!(condition)) { std::cout << message << std::endl; } \
assert ((condition)); } while(false)

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

std::string get_edgetpu_kernel_path(std::string app_name, 
                                    int shape0, 
                                    int shape1){
    std::string path =  "../models/"+ 
                        app_name+"_"+std::to_string(shape0)+"x"+
                                     std::to_string(shape1)+"/"+
                        app_name+"_edgetpu.tflite";
    std::ifstream ifile(path);
    ASSERT_WITH_MESSAGE(ifile.is_open(), 
                        __func__ << ": edgeTPU kernel file: " 
                                 << path << " doesn't exist.");
    return path;
}

void dump_to_csv(std::string log_file_path,
                 std::string app_name,
                 std::string baseline_mode,
                 std::string proposed_mode,
                 unsigned int problem_size,
                 unsigned int block_size,
                 unsigned int iter,
                 Quality* quality,
                 TimeBreakDown* baseline_time_breakdown,
                 TimeBreakDown* proposed_time_breakdown){
    std::fstream myfile;
    // simply append baseline and proposed rows
    myfile.open(log_file_path.c_str(), std::ios_base::app);
    assert(myfile.is_open()); 
    myfile // baseline mode
           << app_name << ","
           << problem_size << ","
           << block_size << ","
           << baseline_mode << ","
           << iter << ","
           << baseline_time_breakdown->input_time_ms << ","
           << baseline_time_breakdown->kernel_time_ms / iter << ","
           << baseline_time_breakdown->output_time_ms << ","
           << baseline_time_breakdown->get_total_time_ms(iter) << std::endl;

           // proposed mode
    myfile << app_name << ","
           << problem_size << ","
           << block_size << ","
           << proposed_mode << ","
           << iter << ","
           << proposed_time_breakdown->input_time_ms << ","
           << proposed_time_breakdown->kernel_time_ms / iter << ","
           << proposed_time_breakdown->output_time_ms << ","
           << proposed_time_breakdown->get_total_time_ms(iter) << ","
           << (baseline_time_breakdown->kernel_time_ms /
                proposed_time_breakdown->kernel_time_ms) << ","
           << (baseline_time_breakdown->get_total_time_ms(iter) /
                proposed_time_breakdown->get_total_time_ms(iter)) << ","
           << quality->rmse(0) / 100 << ","
           << quality->error_rate(0) / 100 << ","
           << quality->error_percentage(0) / 100 << ","
           << quality->ssim(0) << ","
           << quality->pnsr(0) << "," << std::endl;
    myfile.close();
}

