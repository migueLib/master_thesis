# Import
import os
from os.path import abspath, basename
import cv2
import argparse
from argparse import ArgumentParser as AP
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import concurrent.futures
import pandas as pd
import sys

# Local imports
from Fundus import Fundus
from logger import set_logger as sl


def get_args():
    description = """
    Calculates the diameter of folder of Fundus images
    """

    # noinspection PyTypeChecker
    parser = AP(description=description,
                formatter_class=argparse.RawDescriptionHelpFormatter)

    paths = parser.add_argument_group(title="Required Output",
                                      description="Paths")
    paths.add_argument("-s", "--src", dest="src", action="store",
                       required=True, help="Source path")
    paths.add_argument("-o", "--out", dest="out", action="store",
                       required=True, help="Output path")
    arg = parser.parse_args()

    # Standardize paths
    arg.src = abspath(arg.src)
    arg.out = abspath(arg.out)

    return arg


def rgbmean4im_parallel(image):
    """
    Calculates the mean of the R,G,B values of a given Fundus image

    :Arguments:
        image (str): Pathway to the Fundus image

    :Returns:
        image (str): Pathway to the Fundus image
        r (float): The meaned RGB values of the Fundus image
    """
    f = Fundus(image)
    r, g, b = f.get_rgb_mean()
    return image, r, g, b


def main(arg):
    # Make a list with all the images
    logger.info(f"Making list of files with absolute paths")
    logger.info(f"Input data from {arg.src}")
    images = [f"{arg.src}/{f}" for f in os.listdir(arg.src)]

    # Run in parallel
    logger.info(f"Running in parallel")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(rgbmean4im_parallel, images)

    # Output results file
    logger.info(f"Output files to {arg.out}")
    res = [[basename(i), r, g, b] for i, r, g, b in results]
    res_frame = pd.DataFrame(res, columns=["image", "r", "g", "b"])
    res_frame.to_csv(arg.out, index=False)


if __name__ == "__main__":
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
    logger.info(f"Finished")
