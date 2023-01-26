#include <fstream>
#include <iostream>
#include <assert.h>
#include "utils.h"
#include "CH3_pixel_operation.h"

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
    Read a image in file into opencv Mat in CV_8U data type. 
*/
void read_img(const std::string file_name, int rows, int cols, Mat& img){
    Mat raw = imread(file_name);
    assert(!raw.empty());
    cvtColor(raw, img, COLOR_BGR2GRAY);
    resize(img, img, Size(rows, cols), 0, 0, INTER_AREA);
    img.convertTo(img, CV_8U);
}

void mat2array(Mat img, uint8_t* data){
    assert(img.type() % 8 == 0); // CV_8U series
    
    // data has to be pre-allocated with proper size
    if(!img.isContinuous()){
        img = img.clone();
    }
    // row-major
    std::memcpy(data,
                (uint8_t*)img.data,
                img.size().width * img.size().height * sizeof(uint8_t));
}

void mat2array(cuda::GpuMat img, uint8_t* data){
    assert(img.type() % 8 == 0); // CV_8U series
    
    Mat tmp;
    img.download(tmp);
    // data has to be pre-allocated with proper size
    if(!tmp.isContinuous()){
        tmp = tmp.clone();
    }
    // row-major
    std::memcpy(data,
                (uint8_t*)tmp.data,
                tmp.size().width * tmp.size().height * sizeof(uint8_t));
}

void mat2array(Mat img, float* data){
    assert(img.type() % 8 == 5); // CV_32F series
    
    // data has to be pre-allocated with proper size
    if(!img.isContinuous()){
        img = img.clone();
    }
    // row-major
    std::memcpy(data, 
                (float*)img.data, 
                img.size().width * img.size().height * sizeof(float));
}

void mat2array(cuda::GpuMat img, float* data){
    assert(img.type() % 8 == 5); // CV_32F series
    
    Mat tmp;
    img.download(tmp);
    // data has to be pre-allocated with proper size
    if(!tmp.isContinuous()){
        tmp = tmp.clone();
    }
    // row-major
    std::memcpy(data, 
                (float*)tmp.data, 
                tmp.size().width * tmp.size().height * sizeof(float));
}

void array2mat(Mat& img, uint8_t* data, int rows, int cols){
    Mat tmp = Mat(rows, cols, CV_8U, data);
    tmp.copyTo(img);
}

void array2mat(cuda::GpuMat& img, uint8_t* data, int rows, int cols){
    Mat tmp = Mat(rows, cols, CV_8U, data);
    img.upload(tmp);
}

void array2mat(Mat& img, float* data, int rows, int cols){
    Mat tmp = Mat(rows, cols, CV_32F, data);
    tmp.copyTo(img);
}

void array2mat(cuda::GpuMat& img, float* data, int rows, int cols){
    Mat tmp = Mat(rows, cols, CV_32F, data);
    img.upload(tmp);
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

/* plugin function for adjusting laplacian_2d's quality. */
void histogram_matching(void* output_array_baseline,
                        void* output_array_proposed,
                        int rows,
                        int cols,
                        int blk_rows,
                        int blk_cols,
                        std::vector<int> dev_sequence){
    
    std::cout << __func__ << ": testing histogram matching..." << std::endl;
    assert(rows % blk_rows == 0);
    assert(cols % blk_cols == 0);
    unsigned int row_cnt = rows / blk_rows;
    unsigned int col_cnt = cols / blk_cols;
    assert(dev_sequence.size() == (row_cnt * col_cnt));

    /* partition the output and determine which blocks need HM. */
    Mat baseline_mat, proposed_mat;
    Mat baseline_tmp(blk_rows, blk_cols, CV_32F);
    Mat proposed_tmp(blk_rows, blk_cols, CV_32F);
   
    float* baseline_ptr = (float*) malloc( rows * cols * sizeof(float));
    float* proposed_ptr = (float*) malloc( rows * cols * sizeof(float));
    std::memcpy(baseline_ptr, (float*)output_array_baseline, rows * cols * sizeof(float));
    std::memcpy(proposed_ptr, (float*)output_array_proposed, rows * cols * sizeof(float));

    array2mat(baseline_mat, baseline_ptr, rows, cols);
    array2mat(proposed_mat, proposed_ptr, rows, cols);
    
    unsigned int block_total_size = blk_rows * blk_cols;

    // vector of partitions allocation
    //std::vector<void*> proposed_pars;
    //proposed_pars.resize(dev_sequence.size());
    float hm_time_ms = 0.0;

    for(unsigned int i = 0 ; i < row_cnt ; i++){
        for(unsigned int j = 0 ; j < col_cnt ; j++){
            unsigned int idx = i * col_cnt + j;
 
            // partition allocation
            //proposed_pars[idx] = (float*) calloc(block_total_size, sizeof(float));
 
            // partition initialization
            Rect roi(i*blk_rows, j*blk_cols, blk_rows, blk_cols);
            baseline_mat(roi).copyTo(baseline_tmp);
            proposed_mat(roi).copyTo(proposed_tmp);
            baseline_tmp.convertTo(baseline_tmp, CV_8U);
            proposed_tmp.convertTo(proposed_tmp, CV_8U);

            // tiling HM
            std::cout << __func__ << ": dev_sequence[" << idx << "]: " << dev_sequence[idx] <<std::endl;
            if(dev_sequence[idx] == 3){ // tpu
                timing hm_s = clk::now();
                assert(Histst(proposed_tmp, baseline_tmp)); // HS the proposed one based on hist. of baseline.
                timing hm_e = clk::now();
                hm_time_ms += get_time_ms(hm_e, hm_s);
            } // for others, proposed block should remain un-touched.
            proposed_tmp.convertTo(proposed_tmp, CV_32F);
            //mat2array(proposed_tmp, (float*)((proposed_pars[idx])));
        
            //write current block back to out array
            proposed_tmp.copyTo(proposed_mat(roi));
        }
    }
    mat2array(proposed_mat, (float*)output_array_proposed);
    std::cout << __func__ << ": hm time: " << hm_time_ms << " (ms)" << std::endl;
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
                 TimeBreakDown* proposed_time_breakdown,
                 std::vector<int> proposed_device_sequence){
    std::fstream myfile;
    // simply append baseline and proposed rows
    myfile.open(log_file_path.c_str(), std::ios_base::app);
    assert(myfile.is_open());

    auto t = std::chrono::system_clock::now();
    std::time_t ts = std::chrono::system_clock::to_time_t(t);
    std::string ts_str = std::ctime(&ts);

    // log params and performance
    myfile << "*****kernel name*****,problem size,block size,kernel iter,--,--,--,--,--,--,-->,timestamp:,"
           << ts_str
           << app_name << ","
           << problem_size << ","
           << block_size << ","
           << iter << "\n"
           << "*****performance*****,mode,input_conversion ms, kernel avg ms, output_conversion ms,total ms,"
           << "kernel speedup, e2e speedup,,\n"
           << "baseline mode," << baseline_mode << ","
           << baseline_time_breakdown->input_time_ms << ","
           << baseline_time_breakdown->kernel_time_ms / iter << ","
           << baseline_time_breakdown->output_time_ms << ","
           << baseline_time_breakdown->get_total_time_ms(iter) << ",--,--,\n"
           << "proposed mode," << proposed_mode << ","
           << proposed_time_breakdown->input_time_ms << ","
           << proposed_time_breakdown->kernel_time_ms / iter << ","
           << proposed_time_breakdown->output_time_ms << ","
           << proposed_time_breakdown->get_total_time_ms(iter) << ","
           // speedup (proposed over baseline)
           << (baseline_time_breakdown->kernel_time_ms /
                proposed_time_breakdown->kernel_time_ms) << ","
           << (baseline_time_breakdown->get_total_time_ms(iter) /
                proposed_time_breakdown->get_total_time_ms(iter)) << ",\n";
    
    // log quality metrics
    myfile << "*****total quality*****,--,input stats,--,--,--,--,--,output quality,"
           << "\n--,--,max,min,mean,sdev,entropy,--,rmse%,error_rate%,error%,SSIM,PNSR(dB),\n"
           << "--,--," << quality->in_max() << ","
           << quality->in_min() << ","
           << quality->in_mean() << ","
           << quality->in_sdev() << ","
           << quality->in_entropy() << ",,"
           << quality->rmse() << "%,"
           << quality->error_rate() << "%,"
           << quality->error_percentage() << "%,"
           << quality->ssim() << ","
           << quality->pnsr() << ",\n";

    bool is_tiling = (problem_size > block_size)?true:false;
    if(is_tiling){
        myfile << "*****tiling quality*****,(proposed mode's partition),input stats,--,--,--,--,--,output tiling quality,\n"
               << "i,j,max,min,mean,sdev,entropy,device type,rmse%,error_rate%,error%,SSIM,PNSR(dB),\n";
        int idx = 0;
        for(int i = 0 ; i < quality->get_row_cnt() ; i++){
            for(int j = 0 ; j < quality->get_col_cnt() ; j++){
                idx = i * quality->get_col_cnt() +j;
                std::cout << __func__ << " (i: " << i << ", j: " << j << ")" << std::endl;
                myfile << i << "," << j << ","
                       << quality->in_max(i, j) << ","
                       << quality->in_min(i, j) << ","
                       << quality->in_mean(i, j) << ","
                       << quality->in_sdev(i, j) << ","
                       << quality->in_entropy(i, j) << ","
                       << proposed_device_sequence[idx] << ","
                       << quality->rmse(i, j) << "%,"
                       << quality->error_rate(i, j) << "%,"
                       << quality->error_percentage(i, j) << "%,"
                       << quality->ssim(i, j) << ","
                       << quality->pnsr(i, j) << ",\n";
            }
        }
    }
    myfile << "\n";
    myfile.close();
}

