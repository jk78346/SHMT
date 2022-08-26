#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <glob.h>
#include <iostream>

using namespace cv;
using namespace std;

void run_Sobel(std::string input_file_name, std::string out_path){
    Mat image, src, src_gray;
    Mat grad;
    int ddepth = CV_32F;
    image = imread( input_file_name, IMREAD_COLOR);
    if(image.empty()){
        printf("Error opening image: %s\n", input_file_name.c_str());
    }else{
        cvtColor(src, src_gray, COLOR_BGR2GRAY);
        
        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;

        Sobel(src_gray, grad_x, ddepth, 1, 0, 3/*ksize*/, 1/*scale*/, 0/*delta*/, BORDER_DEFAULT);
        Sobel(src_gray, grad_y, ddepth, 0, 1, 3/*ksize*/, 1/*scale*/, 0/*delta*/, BORDER_DEFAULT);

        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);

        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

        imwrite(out_path+, grad);
    }
}

int main(int argc, char** argv){
    glob_t glob_result;
    std::string in_dir = "";
    std::string postfix = "*.jpeg";
    std::string out_dir = in_dir;
    std::string file_name;
    glob((in_dir+postfix).c_str(), GLOB_TILDE, NULL, &glob_result);
    for(int i = 0 ; i < glob_result.gl_pathc; i++){
        file_name = glob_result.gl_pathv[i];
        run_Sobel(file_name, out_dir);
    }

    return 0;
}
