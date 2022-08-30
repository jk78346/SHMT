import glob
import argparse
import cv2 as cv
import numpy as np
import multiprocessing

def get_full_file_name_from_path(file_path):
    return file_path.split("/")[-1]

def get_file_name_wo_ext_from_path(file_path):
    file_full_name = get_full_file_name_from_path(file_path)
    return file_full_name.split(".")[0]

def generate_resized_img(file_path, size, resized_in_dir):
    gray = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    if gray is None:
        print("Error opening image: ", file_path)
        exit(0)
    resized = cv.resize(gray, (size, size), interpolation=cv.INTER_NEAREST)
    out = np.asarray(resized)
    resized_in_path = resized_in_dir + get_file_name_wo_ext_from_path(file_path)
    np.save(resized_in_path, out)
    return resized_in_path, resized

def run_Sobel(input_file_path, src, size, out_path, postfix, file_ext):
    ddepth = cv.CV_32F
    grad_x = cv.Sobel(src, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(src, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    out = np.asarray(grad)
    out_file_name = out_path+"/"+get_file_name_wo_ext_from_path(input_file_path) + postfix
    np.save(out_file_name, out)

def main():
    parser = argparse.ArgumentParser(description='Sobel operation on dataset')
    parser.add_argument(dest='in_dir', action='store', type=str, help='input dir')
    parser.add_argument(dest='resized_in_dir', action='store', type=str, help='resized_input dir')
    parser.add_argument(dest='out_dir', action='store', type=str, help='output dir')
    parser.add_argument(dest='target_size', action='store', type=int, help='target_size')
    args = parser.parse_args()
    in_dir = args.in_dir
    resized_in_dir = args.resized_in_dir
    out_dir = args.out_dir
    target_size = args.target_size
    postfix = "_Sobel"
    file_ext = "JPEG"
    print(args)
    num_in_dir_files = len(glob.glob(in_dir+"/*"))
    for idx, fpath in enumerate(glob.glob(in_dir+"/*")):
        resized_in_path, resized = generate_resized_img(fpath, target_size, resized_in_dir)
        run_Sobel(resized_in_path, resized, target_size, out_dir, postfix, file_ext)
        print("Sobel progress: ", idx, "/", num_in_dir_files, ", resized_input_file: ", resized_in_path)

if __name__ == "__main__":
    main()
