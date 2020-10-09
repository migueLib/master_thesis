import cv2, glob, numpy
import os
from os.path import abspath
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.image as mpimg
from skimage.transform import resize
from PIL import Image
import argparse
from argparse import ArgumentParser as AP
from tqdm import tqdm
from pathlib import Path


def get_args():
    description = """
    Normalizes images using BennGraham code
    """
    
    # noinspection PyTypeChecker
    parser = AP(description=description,
               formatter_class=argparse.RawDescriptionHelpFormatter)
    
    paths = parser.add_argument_group(title="Required Output",
                                     description="Paths")
    paths.add_argument("-s", "--src", dest="src", action="store",
                      required=True, help="Source path")
    paths.add_argument("-d", "--dst", dest="dst", action="store",
                      required=True, help="Destination path")
    arg = parser.parse_args()
    
    # Standardize paths
    arg.src = abspath(arg.src)
    arg.dst = abspath(arg.dst)
    
    return arg


def is_image(f):
    """
    Checks if file is image
    """
    flag = False
    
    if f.endswith('.jpg'):
        flag = True
    elif f.endswith('.JPG'):
        flag = True
    elif f.endswith('.png'):
        flag = True
    elif f.endswith('.PNG'):
        flag = True
    elif f.endswith('.jpeg'):
        flag = True
    elif f.endswith('.JPEG'):
        flag = True
    elif f.endswith('.tif'):
        flag = True
    else: 
        flag = False

    return flag


def normalize_image(image):
    """
    Normalize image 
    """
    gaussian_blur = cv2.GaussianBlur(image, (0, 0), 1000/30)
    normalized_im = cv2.addWeighted(image, 4, gaussian_blur, -4, 128)
    
    return normalized_im

   

def normalize_folder(in_folder, out_folder):
    """
    Normalize folder image
    """
    path = str(in_folder) + "/**"
    
    # Make directory if it does not exist
    Path(out_folder).mkdir(parents=True, exist_ok=True)
        
    # Iterate over the source folder and normalize the images
    for f in tqdm(glob.glob(path, recursive=True), leave=True):
        
        if is_image(f):
            # Original image
            ori_img = cv2.imread(f)

            # Normalize image
            normalized_im = normalize_image(ori_img)

            # normalization according to the instructions
            out_name = os.path.join(str(out_folder), os.path.basename(f))
            
            # Save image and prints a message if not safe 
            if not cv2.imwrite(f"{out_name[:-4]}.png", normalized_im):
                print(f"{f} was not normalized")

        else:
            print(f"{f} is not an Image")

            
def main(arg):
    # Normalize images
    normalize_folder(arg.src, arg.dst)

    
if __name__ == "__main__":
    # Stablish working directory
    os.chdir(os.path.dirname(os.getcwd()))
    
    # Load arguments
    args = get_args()

    # Set logger
    # logger = sl("info")
    
    # Initial Report
    # logger.info(f"Python version {sys.version}")
    # logger.info(f"Working directory {os.getcwd()}")
    # logger.info(f"Usable CPUs {os.cpu_count()}")
    
    # Main
    main(args)