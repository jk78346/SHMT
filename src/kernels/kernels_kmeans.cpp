#include "kernels_cpu.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

cv::Mat vectorQuantization(cv::Mat vec,cv::Mat centers){
	// accept only char type matrices
	CV_Assert(vec.depth() != sizeof(uchar));
	
	int rows=vec.rows;
	int cols=vec.cols;
	cv::Mat quantize(rows,centers.cols,centers.type());
	uchar *q=quantize.ptr<uchar>(0);
	uchar *v=vec.ptr<uchar>(0);
	uchar *c=centers.ptr<uchar>(0);
	for(int i=0;i<rows;i++){
	
		q[3*i]=c[3*v[i]];
		q[3*i+1]=c[3*v[i]+1];
		q[3*i+2]=c[3*v[i]+1];
	
	}
	return quantize;
}

void CpuKernel::kmeans_2d(const Mat in_img, Mat& out_img){
    int k = 2;
    cv::Mat labels, centers;
    cv::Mat src_img = cv::imread("../data/lena.png");
    cv::Mat src_reshaped = src_img.reshape(1, src_img.rows * src_img.cols);
    src_reshaped.convertTo(src_reshaped, CV_32FC1);
    cv::kmeans(src_reshaped, 
               k, 
               labels, 
               cv::TermCriteria(TermCriteria::MAX_ITER|TermCriteria::EPS, 
                                10, // max iteration 
                                1.0), // epsilon
               3, // attempts
               cv::KMEANS_RANDOM_CENTERS,
               centers);

    centers.convertTo(centers,CV_8UC1);
	labels.convertTo(labels,CV_8UC1);

	cv::Mat quantized=vectorQuantization(labels,centers);
	
	quantized=quantized.reshape(3,src_img.rows);
	quantized.convertTo(quantized,CV_8UC3);
    cv::imwrite("../data/lena_kmeans_"+std::to_string(k)+".png", quantized);
}
