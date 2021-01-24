import cv2, glob
import numpy as np
import os
from os.path import abspath
import matplotlib.pyplot as plt
from multiprocessing import Pool
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

def get_radius(img):
    """
    Gets the radius of a Fundus photograph at the x axis
    """
    # Gets the pixels in the midle (vertically) of the Fundus
    # Then sums their RGB values
    x = img[img.shape[0]//2, : , :].sum(1)
    
    # Use 1/10 of the mean as a threshold, 
    # anything above count it as part of the eye
    # This value is the radius
    r = (x > x.mean()/10).sum()/2
    
    return r


def normalize_image(img):
    """
    Normalizes image 
    """
    # Get radius of the image
    r = get_radius(img)
    
    # This will ensure that even if a figure contains no fundus (or is black) the script won't stop
    r = 690 if r==0 else r 
    
    # Calculate the Gaussian Blur based on the radius
    gaussian_blur = cv2.GaussianBlur(src=img, ksize=(0, 0), sigmaX=r/30)
    
    # Blend GB and Original Image
    normalized_im = cv2.addWeighted(src1=img, alpha=4, src2=gaussian_blur, beta=-4, gamma=128)
    
    # Create a circular mask to remove the outter 1%
    # (This will avoid frontier effects)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.circle(img=mask, center=(img.shape[1]//2, img.shape[0]//2), radius=int(r*0.99), color=(1,1,1,1), thickness=-1, lineType=8, shift=0)
    
    # Apply mask
    normalized_im = cv2.bitwise_and(normalized_im, normalized_im, mask=mask)
    
    return normalized_im

            
def normalize_folder(in_folder, out_folder):
    """
    Normalize folder
    """
    for r, d, files in os.walk(in_folder):

        # Recreate structure from input folder
        r_i = Path(*Path(r).parts[len(Path(in_folder).parts):])
        r_o = Path(os.path.join(out_folder, r_i))
        r_o.mkdir(parents=True, exist_ok=True)

        # Iterate over each file
        for f in tqdm(files, leave=True):

            # Check if file is an image
            if is_image(f):

                # Load image
                ori_img = cv2.imread(os.path.join(r, f))

                # Normalize image
                nor_img = normalize_image(ori_img)

                # Save image and print a message if the image was not saved
                if not cv2.imwrite(f"{os.path.join(r_o, f)[:-4]}.png", nor_img):
                    print(f"{f} was not normalzied")

            else:
                print(f"{f} is not an image")
            
def main(arg):
    # Normalize images
    normalize_folder(arg.src, arg.dst)

    
if __name__ == "__main__":
    # Stablish working directory
    os.chdir(os.path.dirname(os.getcwd()))
    
    # Load arguments
    args = get_args()

    # Set logger
    logger = sl("info")
    
    # Initial Report
    logger.info(f"Python version {sys.version}")
    logger.info(f"Working directory {os.getcwd()}")
    logger.info(f"Usable CPUs {os.cpu_count()}")
    
    # Main
    main(args)