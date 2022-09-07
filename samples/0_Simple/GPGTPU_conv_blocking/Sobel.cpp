#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <glob.h>
#include <omp.h>
#include <iostream>

using namespace cv;
using namespace std;

void get_full_file_name_from_path(string file_path, string& file_name){
    file_name = file_path.substr(file_path.find_last_of("/\\")+1);
}

void get_file_name_wo_ext_from_path(string file_path, string& file_name){
    string base_file_name;
    get_full_file_name_from_path(file_path, base_file_name);
    string::size_type const p(base_file_name.find_last_of('.'));
    file_name = base_file_name.substr(0, p);
}

void generate_resized_img(string file_path, int size, string resized_in_dir, string& resized_in_path, vector<int> params){
    Mat image, src_gray;
    image = imread(file_path);
    if(image.empty()){
        printf("Error opening image: %s\n", file_path.c_str());
    }else{
        cvtColor(image, src_gray, COLOR_BGR2GRAY);
        resize(src_gray, src_gray, Size(size, size), 0, 0, INTER_NEAREST);
        string file_name;
        get_full_file_name_from_path(file_path, file_name); 
        resized_in_path = resized_in_dir+file_name;
        imwrite(resized_in_path, src_gray, params);
    }
}

void run_Sobel(string input_file_path, int size, string out_path, string postfix, string file_ext, vector<int> params){
    Mat image, src_gray;
    Mat grad;
    int ddepth = CV_32F;
    image = imread(input_file_path);
    if(image.empty()){
        printf("Error opening image: %s\n", input_file_path.c_str());
    }else{
        // ===== Sobel main process =====
        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;

        Sobel(image, grad_x, ddepth, 1, 0, 3/*ksize*/, 1/*scale*/, 0/*delta*/, BORDER_DEFAULT);
        Sobel(image, grad_y, ddepth, 0, 1, 3/*ksize*/, 1/*scale*/, 0/*delta*/, BORDER_DEFAULT);

        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);

        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

        // give output file name
        string input_file_name_prefix;

        get_file_name_wo_ext_from_path(input_file_path, input_file_name_prefix);

        string out_file_name = out_path + "/" + input_file_name_prefix + postfix + "." + file_ext;
        imwrite(out_file_name, grad, params);
    }
}

int main(int argc, char** argv){
    if(argc != 5){
        cout << "Usage: " << argv[0] << " <r:in_dir> <w:resized_in_dir> <w:out_dir> <target_size>" << endl;
        exit(0);
    }
    string in_dir         = argv[1];
    string resized_in_dir = argv[2]; 
    string out_dir        = argv[3]; 
    int size              = atoi(argv[4]);
    string file_ext       = "JPEG";
    string postfix        = "_Sobel";
    string in_file_re     = "/*."+file_ext;
    
    // Support for writing jpg
    const int JPEG_QUALITY = 95;
    vector<int> params;
    params.push_back(IMWRITE_JPEG_QUALITY);
    params.push_back(JPEG_QUALITY);
    
    glob_t glob_result;
    cout << "globing files from " <<  in_dir+in_file_re << "...";
    glob((in_dir+in_file_re).c_str(), GLOB_TILDE, NULL, &glob_result);
    cout << ", and " << glob_result.gl_pathc << " files found." << endl;

    long long int global_cnt = 0;
    
    #pragma omp parallel for num_threads(8)
    for(int i = 0 ; i < glob_result.gl_pathc; i++){
        string file_path = glob_result.gl_pathv[i];
        string resized_in_path;

        generate_resized_img(file_path, size, resized_in_dir, resized_in_path, params);
        
        run_Sobel(resized_in_path, size, out_dir, postfix, file_ext, params);

        #pragma omp atomic update
        global_cnt+=1;
        
        #pragma omp critical
        {
            cout << "Sobel progress: (" << global_cnt+1 << "/" << glob_result.gl_pathc << "), resized_input_file: " << resized_in_path.c_str() << endl;
        }
    }       
    return 0;
}
