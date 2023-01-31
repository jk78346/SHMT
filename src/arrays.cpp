#include <string>
#include <opencv2/opencv.hpp>
#include "arrays.h"
#include "utils.h"
#include "Common.h"
#include "BmpUtil.h"

/*
    Input array allocation and initialization.
    The data type of arrays depends on applications.
    EX:
        uint8_t: sobel_2d, mean_2d, laplacian_2d
        float:   fft_2d, dct8x8_2d, blackscholes
 */

std::vector<std::string> uint8_t_type_app = {
    "sobel_2d",
    "mean_2d",
    "laplacian_2d",
    "kmeans_2d"
};

void init_fft(unsigned int input_total_size, void** input_array){
    printf("...generating random input data\n");
    srand(2010);
    float* tmp = reinterpret_cast<float*>(*input_array);
    for(unsigned int i = 0 ; i < input_total_size ; i++){
        tmp[i] = (float)(rand() % 16);        
    }
}

void init_dct8x8(int rows, int cols, void** input_array){
    /* Reference: samples/3_Imaging/dct8x8/dct8x8.cu */
    char SampleImageFname[] = "../data/barbara.bmp";
    char *pSampleImageFpath = sdkFindFilePath(SampleImageFname, NULL/*argv[0]*/);
    if (pSampleImageFpath == NULL)
    {
        printf("dct8x8 could not locate Sample Image <%s>\nExiting...\n", pSampleImageFpath);
        exit(EXIT_FAILURE);
    }

    //preload image (acquire dimensions)
    int ImgWidth = rows;
    int ImgHeight = cols;
    ROI ImgSize;
    int res = PreLoadBmp(pSampleImageFpath, &ImgWidth, &ImgHeight);
    ImgSize.width = ImgWidth;
    ImgSize.height = ImgHeight;
    if (res)
    {
        printf("\nError: Image file not found or invalid!\n");
        exit(EXIT_FAILURE);
    }

    //check image dimensions are multiples of BLOCK_SIZE
    if (ImgWidth % BLOCK_SIZE != 0 || ImgHeight % BLOCK_SIZE != 0)
    {
        printf("\nError: Input image dimensions must be multiples of 8!\n");
        exit(EXIT_FAILURE);
    }

    //printf("[%d x %d]... ", ImgWidth, ImgHeight);

    //allocate image buffers
    int ImgStride;
    byte *ImgSrc = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
    //load sample image
    LoadBmpAsGray(pSampleImageFpath, ImgStride, ImgSize, ImgSrc);

    /* ImgSrc has to be resized to [rows x cols] */
    // byte = unsigned char
    byte *ImgSrc_resized = MallocPlaneByte(rows, cols, &ImgStride);
    Mat tmp, resized_tmp;
    array2mat(tmp, ImgSrc, rows, cols);
    Size size = Size(rows, cols);
    resize(tmp, resized_tmp, size);
    mat2array(resized_tmp, ImgSrc_resized);
    ImgSize.width = rows;
    ImgSize.height = cols;

    /* Reference: samples/3_Imaging/dct8x8/dct8x8.cu: float WrapperCUDA2() function */   
    //allocate host buffers for DCT and other data
    int StrideF;
    float *ImgF1 = MallocPlaneFloat(rows, cols, &StrideF);

    //convert source image to float representation
    CopyByte2Float(ImgSrc_resized, ImgStride, ImgF1, StrideF, ImgSize);
    AddFloatPlane(-128.0f, ImgF1, StrideF, ImgSize);
    
    *input_array = ImgF1;
}

void read_hotspot_file(float* vect, int grid_rows, int grid_cols, const char* file){
    int i;//, index;
    FILE *fp;
    int STR_SIZE = 256;
    char str[STR_SIZE];
    float val;
     
    fp = fopen (file, "r");
    if (!fp){
        std::cout << __func__ << ": file could not be opened for reading" << std::endl;
        exit(0);
    }
    for (i=0; i < grid_rows * grid_cols; i++) {
        fgets(str, STR_SIZE, fp);
        if (feof(fp)){
            std::cout << __func__ << ": not enough lines in file" << std::endl;
            exit(0);
        }
        if ((sscanf(str, "%f", &val) != 1) ){
            std::cout << __func__ << ": invalid file format" << std::endl;
            exit(0);
        }
        vect[i] = val;
    }
     
    fclose(fp);
}

void init_hotspot(int rows, int cols, void** input_array){
    float* float_ptr = reinterpret_cast<float*>(*input_array);

    std::string tfile = "../data/hotspot/temp_"+std::to_string(rows);
    std::string pfile = "../data/hotspot/power_"+std::to_string(rows);

    read_hotspot_file(float_ptr, rows, cols, tfile.c_str());
    int offset = rows * cols;
    // concate temp and power arrays into input_array
    read_hotspot_file(&float_ptr[offset], rows, cols, pfile.c_str());

    float temp_max = FLT_MIN;
    float temp_min = FLT_MAX;
    float power_max = FLT_MIN;
    float power_min = FLT_MAX;

    float power_sum = 0.0;

    for(int i = 0 ; i < rows * cols ; i++){
        if(float_ptr[i] > temp_max){
            temp_max = float_ptr[i];
        }
        if(float_ptr[i] < temp_min){
            temp_min = float_ptr[i];
        }
        if(float_ptr[i+offset] > power_max){
            power_max = float_ptr[i+offset];
        }
        if(float_ptr[i+offset] < power_min){
            power_min = float_ptr[i+offset];
        }
        power_sum += float_ptr[i+offset];
    }
    std::cout << __func__ << ": temp max: " << temp_max
                          << ", min: " << temp_min << std::endl;
    std::cout << __func__ << ": power max: " << power_max
                          << ", min: " << power_min << std::endl;
    std::cout << __func__ << ": power sum: " << power_sum << std::endl;
}

void data_initialization(Params params,
                         void** input_array,
                         void** output_array_baseline,
                         void** output_array_proposed){
    // TODO: choose corresponding initial depending on app_name.
    int rows = params.problem_size;
    int cols = params.problem_size;
    unsigned int input_total_size = rows * cols;
    unsigned int output_total_size = rows * cols;

    // image filter type of kernels
    if( std::find(uint8_t_type_app.begin(), 
                  uint8_t_type_app.end(), 
                  params.app_name) !=
        uint8_t_type_app.end() ){
        *input_array = (uint8_t*) malloc(input_total_size * sizeof(uint8_t));
        *output_array_baseline = (uint8_t*) malloc(output_total_size * sizeof(uint8_t));
        *output_array_proposed = (uint8_t*) malloc(output_total_size * sizeof(uint8_t));        
        Mat in_img;
        read_img(params.input_data_path,
                 rows,
                 cols,
                 in_img);
        mat2array(in_img, (uint8_t*)*input_array);
                
        /* extreme data partittion distribution experiment. */
        //std::cout << __func__ << ": params - problem_size: " << params.problem_size 
        //          << ", block_size: " << params.block_size
        //          << ", get_row_cnt(): " << params.get_row_cnt()
        //          << ", get_col_cnt(): " << params.get_col_cnt() << std::endl;
        //unsigned int row_cnt = params.problem_size / params.block_size;
        //unsigned int col_cnt = params.problem_size / params.block_size;
        //uint8_t* tmp = reinterpret_cast<uint8_t*>(*input_array);
        //for(unsigned int i = 0 ; i < row_cnt; i++){
        //    for(unsigned int j = 0 ; j < col_cnt ; j++){
        //        int step = 256 / (row_cnt * col_cnt);
        //        uint8_t max = (i*col_cnt+j)*step+1;
        //        for(unsigned int ii= 0 ; ii < params.block_size ; ii++){
        //            for(unsigned int jj = 0 ; jj < params.block_size ; jj++){
        //                tmp[(i*params.block_size+ii)*params.problem_size+(j*params.block_size+jj)] = rand() % max; //rand()%10 + max;
        //            }
        //        }
        //    }
        //}
    }else{ // others are default as float type
        *output_array_baseline = (float*) malloc(output_total_size * sizeof(float));
        *output_array_proposed = (float*) malloc(output_total_size * sizeof(float));   
        if(params.app_name == "fft_2d"){
            *input_array = (float*) malloc(input_total_size * sizeof(float));
            init_fft(input_total_size, input_array);
        }else if(params.app_name == "dct8x8_2d"){
            init_dct8x8(rows, cols, input_array);
        }else if(params.app_name == "hotspot_2d"){
            *input_array = (float*) malloc(2 * input_total_size * sizeof(float));   
            init_hotspot(rows, cols, input_array);
        }else{
            *input_array = (float*) malloc(input_total_size * sizeof(float));
            Mat in_img;
            read_img(params.input_data_path,
                    rows,
                    cols,
                    in_img);
            in_img.convertTo(in_img, CV_32F);
            mat2array(in_img, (float*)*input_array);
        }
    }
}

/*
    partition array into partitions: allocation and initialization(optional).
*/
void array_partition_initialization(Params params,
                                    bool skip_init,
                                    void** input,
                                    std::vector<void*>& input_pars){
    if( std::find(uint8_t_type_app.begin(), 
                  uint8_t_type_app.end(), 
                  params.app_name) !=
        uint8_t_type_app.end() ){
        // prepare for utilizing opencv roi() to do partitioning.
        Mat input_mat, tmp(params.block_size, params.block_size, CV_8U);
        if(!skip_init){
            array2mat(input_mat, (uint8_t*)*input, params.problem_size, params.problem_size);
        }
        unsigned int block_total_size = params.block_size * params.block_size;

        // vector of partitions allocation
        input_pars.resize(params.get_block_cnt());   
        for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
            for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){ 
                unsigned int idx = i * params.get_col_cnt() + j;         
        
                // partition allocation
                input_pars[idx] = (uint8_t*) calloc(block_total_size, sizeof(uint8_t));

                // partition initialization
                if(!skip_init){
                    Rect roi(i*params.block_size, j*params.block_size, params.block_size, params.block_size); 
                    input_mat(roi).copyTo(tmp); 
                    mat2array(tmp, (uint8_t*)((input_pars[idx])));
                }
            }
        }
    }else{
        // prepare for utilizing opencv roi() to do partitioning.
        Mat input_mat, tmp(params.block_size, params.block_size, CV_32F);
        if(!skip_init){
            array2mat(input_mat, (float*)*input, params.problem_size, params.problem_size);
        }
        unsigned int block_total_size = params.block_size * params.block_size;

        // vector of partitions allocation
        input_pars.resize(params.get_block_cnt());   
        for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
            for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){ 
                unsigned int idx = i * params.get_col_cnt() + j;         
        
                // partition allocation
                input_pars[idx] = (float*) calloc(block_total_size, sizeof(float));
                
                // partition initialization
                if(!skip_init){
                    Rect roi(i*params.block_size, j*params.block_size, params.block_size, params.block_size); 
                    input_mat(roi).copyTo(tmp); 
                    mat2array(tmp, (float*)((input_pars[idx])));
                }
            }
        }
    }
}

/*
    Remap output partitions into one single output array.
*/
void output_array_partition_gathering(Params params,
                                      void** output,
                                      std::vector<void*>& output_pars){
    // prepare for utilizing opencv roi() to do gathering.
    if( std::find(uint8_t_type_app.begin(), 
                  uint8_t_type_app.end(), 
                  params.app_name) !=
        uint8_t_type_app.end() ){
        Mat output_mat(params.problem_size, params.problem_size, CV_8U), tmp;
    
        for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
            for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){
                unsigned int idx = i * params.get_col_cnt() + j;
                array2mat(tmp, (uint8_t*)((output_pars[idx])), params.block_size, params.block_size);
                Rect roi(i*params.block_size, j*params.block_size, params.block_size, params.block_size); 
                tmp.copyTo(output_mat(roi));
            }
        }
        mat2array(output_mat, (uint8_t*)*output);
    }else{
        Mat output_mat(params.problem_size, params.problem_size, CV_32F), tmp;
    
        for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
            for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){
                unsigned int idx = i * params.get_col_cnt() + j;
                array2mat(tmp, (float*)((output_pars[idx])), params.block_size, params.block_size);
                Rect roi(i*params.block_size, j*params.block_size, params.block_size, params.block_size); 
                tmp.copyTo(output_mat(roi));
            }
        }
        mat2array(output_mat, (float*)*output);
    }    
}

void sampling_kernel(Params params, Mat& in, Mat& out, int downsample_block_size){
    auto mode = params.get_sampling_mode();
    if(mode == cv_resize){
        cv::resize(in,
                   out, 
                   cv::Size(downsample_block_size, downsample_block_size), 
                   0, 
                   0, 
                   cv::INTER_LINEAR);
    
    }else{ // other cropping types
        int i_start = 0;
        int j_start = 0;
        if(mode == init_crop){
            i_start = j_start = 0;
        }else if(mode == center_crop){ // assume both square sizes
            i_start = j_start = (in.rows- downsample_block_size)/2;
        }else if(mode == random_crop){
            i_start = int(rand() % (in.rows - downsample_block_size));
            j_start = int(rand() % (in.rows - downsample_block_size));
        }
        Rect roi(i_start, j_start, downsample_block_size, downsample_block_size); 
        in(roi).copyTo(out);
    }
}

void array_partition_downsampling(Params params,
                                  bool skip_init,
                                  std::vector<void*> input_pars,
                                  std::vector<void*>& input_sampling_pars){
    float rate = params.get_downsampling_rate();
    if( std::find(uint8_t_type_app.begin(), 
                  uint8_t_type_app.end(), 
                  params.app_name) !=
        uint8_t_type_app.end() ){

        input_sampling_pars.resize(params.get_block_cnt());

        for(unsigned int i = 0 ; i < params.get_row_cnt() ; i++){
            for(unsigned int j = 0 ; j < params.get_col_cnt() ; j++){
                unsigned int idx = i * params.get_col_cnt() + j;

                unsigned int downsample_block_size = params.block_size * rate;
                unsigned int block_total_size = 
                    downsample_block_size * downsample_block_size;

                // downsampling partition allocation
                input_sampling_pars[idx] = 
                    (uint8_t*) calloc(block_total_size, sizeof(uint8_t));
                
                if(!skip_init){
                    Mat tmp, sampling_tmp;
                    array2mat(tmp,
                              (uint8_t*)input_pars[idx],
                              params.block_size,
                              params.block_size);
                    array2mat(sampling_tmp, 
                              (uint8_t*)input_sampling_pars[idx], 
                              downsample_block_size, 
                              downsample_block_size);
        
                    // actual downsampling
                    sampling_kernel(params, tmp, sampling_tmp, downsample_block_size);
                
                    // store back to input_sampling_pars[idx]
                    mat2array(sampling_tmp, (uint8_t*)input_sampling_pars[idx]);
                }
            }
        }
    }else{
        std::cout << __func__ 
                  << ": downsampling on float type is not implemented yet." 
                  << std::endl;
    }
}

