#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <glob.h>
#include <iostream>

using namespace cv;
using namespace std;

void run_Sobel(string input_file_name, string out_path, string postfix, string file_ext){
    Mat image, src, src_gray;
    Mat grad;
    int ddepth = CV_32F;
    image = imread( input_file_name);
    cout << "image.size: " << image.size() << endl;
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

        // give output file name
        string input_file_name_prefix = input_file_name.substr(0, input_file_name.find("."));
        string out_file_name = out_path + "/" + input_file_name_prefix + postfix + "." + file_ext;

        imwrite(out_file_name, grad);
    }
}

int main(int argc, char** argv){
    if(argc != 3){
        cout << "Usage: " << argv[0] << " <in_dir> <out_dir>" << endl;
        exit(0);
    }
    string in_dir     = argv[1];
    string out_dir    = argv[2];
    string file_ext   = "JPEG";
    string postfix    = "_Sobel";
    string in_file_re = "/*."+file_ext;
    
    glob_t glob_result;
    glob((in_dir+in_file_re).c_str(), GLOB_TILDE, NULL, &glob_result);
    
    for(int i = 0 ; i < glob_result.gl_pathc; i++){
        cout << "Sobel progress: " << (i/glob_result.gl_pathc)*100 << " %, in_dir: " << in_dir.c_str() << endl;
        string file_name = glob_result.gl_pathv[i];
        run_Sobel(file_name, out_dir, postfix, file_ext);
    }

    return 0;
}
