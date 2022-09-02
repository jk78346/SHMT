import os
import subprocess

def get_gittop():
    """ This function returns the absolute path of current git repo root. """
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], \
                            stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')

def get_imgs_count(path, ext):
    """ This function returns number of 'ext' type of files under 'path' dir. """
    return len([os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(ext)])

def get_img_paths_list(path, ext, sort=True):
    """ This function returns list of 'ext' type of files under 'path' dir. """
    ret = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(ext)]
    return sorted(ret) if sort == True else ret




