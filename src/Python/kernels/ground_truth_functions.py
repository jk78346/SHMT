import scipy
import cv2 as cv
import numpy as np
from scipy.fftpack import dct, idct

class Applications:
    """ This class provides a series of application functions as ground truth. """
    def __init__(self, func_name):
        self.func_name = func_name

    def get_func(self):
        """ Return corresponding application function function object """
        func = getattr(self, self.func_name)
        assert callable(func), \
            f" Application name: {func_name} not found. "
        return func

    @staticmethod
    def minimum_2d(src):
        """ This function returns a minimum kernel of given input shape. """
        assert(len(src.shape) == 2) ,\
                f" minimum_2d: # of dims of input != 2, found {len(src.shape)}. "
        # quality result can be ignored, this kernel is latency unity test only
        return src 

    @staticmethod
    def sobel_2d(src):
        """ This function returns edge detected 2D image utilizing OpenCV Sobel filters. """
        assert(len(src.shape) == 2) ,\
                f" sobel_2d: # of dims of input != 2, found {len(src.shape)}. "
        ddepth = cv.CV_32F
        grad_x = cv.Sobel(src, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(src, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return np.asarray(grad)
    
    @staticmethod
    def npu_sobel_2d(src):
        """ This function returns edge detected 2D image utilizing OpenCV Sobel filters. """
        assert(len(src.shape) == 2) ,\
                f" sobel_2d: # of dims of input != 2, found {len(src.shape)}. "
        ddepth = cv.CV_32F
        grad_x = cv.Sobel(src, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(src, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return np.asarray(grad)

    @staticmethod
    def mean_2d(src):
        """ mean filter on 2D image """
        assert(len(src.shape) == 2), \
            f" mean_2d: # of dims of input != 2, found {len(src.shape)}. "
        blur = cv.blur(src, (3, 3), borderType=cv.BORDER_DEFAULT)
        return blur

    @staticmethod
    def laplacian_2d(src):
        """ Laplacian operator on 2D image """
        assert(len(src.shape) == 2), \
            f" mean_2d: # of dims of input != 2, found {len(src.shape)}. "
        ddepth = cv.CV_32F
        ret = cv.Laplacian(src, ddepth, ksize=3)
        ret = cv.convertScaleAbs(ret)
        return ret

    @staticmethod
    def fft_2d(src, kernel, kernelY=3, kernelX=4):
        """ This function implements a R2c / C2R FFT-based convolution that \
            mimic the behavior as the example code in \
            GPGTPU/samples/3_Imaging/convolutionFFT2D """
        dataH, dataW = src.shape
        kernelH, kenrelW = kernel.shape  # 7, 6
        ret = scipy.signal.fftconvolve(src, kernel, mode='same')
        return ret

    @staticmethod
    def dct8x8_2d(src):
        """ This function implements dct8x8. """
        imF = dct(dct(src.T, norm='ortho').T, norm='ortho')
        return imF
        #return idct(idct(imF.T, norm='ortho').T, norm='ortho')

    @staticmethod
    def histogram256(src):
        """ This function returns historgram 256 of a array. 
            cv.calHist returns an array of histogram points of dtype float.32
            , while every src.size chunk of the array is a 8 bit chunk of output int32 result. 
        """
        hist = cv.calcHist([src], [0], None, [256], (0, 256), accumulate=False)
        hist = hist.astype(np.uint32)
        x0 = hist
        x1 = hist >> 8
        x2 = hist >> 16
        x3 = hist >> 24

        x0 = np.remainder(x0, 256)
        x1 = np.remainder(x1, 256)
        x2 = np.remainder(x2, 256)
        x3 = np.remainder(x3, 256)

        ret = np.concatenate((x0, x1, x2, x3), axis=None)
        return ret






