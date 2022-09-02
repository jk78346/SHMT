import cv2 as cv
import numpy as np

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
