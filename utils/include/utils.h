#include <opencv2/opencv.hpp>

using namespace cv;

class Utils{
	public:
		void mat2array(Mat& img, float* data);
    		void array2mat(Mat& img, float* data, int CV_type, int rows, int cols);	
};
