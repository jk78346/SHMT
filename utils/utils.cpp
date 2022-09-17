#include "utils.h"

void Utils::mat2array(Mat& img, float* data){
	// data has to be pre-allocated with proper size
	if(! img.isContinuous()){
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

void Utils::array2mat(Mat& img, float* data, int CV_type, int rows, int cols){
	Mat tmp = Mat(rows, cols, CV_type, data);
	tmp.copyTo(img);
}

