import cv2
import os
import sys
from os.path import abspath, isdir, isfile, basename
import argparse
from argparse import ArgumentParser as ArP
from pathlib import Path

# Local imports
from logger import set_logger as sl
from Fundus import Fundus


def get_args():
    description = """
    Normalizes images using BennGraham code
    """

    # noinspection PyTypeChecker
    parser = ArP(description=description,
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


def from_folder(src: str, dst: str) -> None:
    """
    Normalizes images from a src directory and outputs the normalized files onto
    an dst directory. The dst directory will have the same structure as the src directory.

    Args:
        src (str): Pathway to Fundus images to normalize.
        dst (str): Pathway to output normalized Fundus images.
    """
    for r, d, files in os.walk(src):

        # Recreate structure from input folder
        r_i = Path(*Path(r).parts[len(Path(src).parts):])
        r_o = Path(os.path.join(dst, r_i))
        r_o.mkdir(parents=True, exist_ok=True)

        # Iterate over each file
        for f in files:
            
            # Load image
            original = Fundus(os.path.join(r, f))

            # Normalize image
            nor_img = original.normalize()
                
            # Save image and print a message if the image was not saved
            if not cv2.imwrite(f"{os.path.join(r_o, f)[:-4]}.png", nor_img):
                print(f"{f} was not normalized")


def from_file(src: str, dst: str) -> None:
    """
    Normalizes images from a src file with paths to images
    and outputs the normalized results to a dst directory.

    The input src file must not contain headers or other columns.

    path/to/file1.png
    path/to/file2.png
    ...

    Args:
        src (str): Path to csv file with Fundus images paths
        dst (str): Path to output results
    """

    # Creates output folder if doesn't exist
    output_folder = Path(os.path.join(dst))
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load the file names of the image to normalize from src
    with open(src, "r") as FILES:
        files = [f.strip() for f in FILES.readlines()]

    # Iterate over files
    for f in files:
        # Load image
        original = Fundus(f)

        # Normalize image
        nor_img = original.normalize()

        # Save image and print a message if the image was not saved
        if not cv2.imwrite(f"{os.path.join(output_folder, basename(f))[:-4]}.png", nor_img):
            print(f"{f} was not normalized")


def main(arg):
    # Normalize images
    logger.info(f"Loading data from {arg.src}")
    logger.info(f"Saving data to {arg.dst}")

    # Checking if src is file o folder
    if isdir(arg.src):
        logger.info(f"Iterating from folder")
        from_folder(arg.src, arg.dst)
    elif isfile(arg.src):
        logger.info(f"Loading from file")
        from_file(arg.src, arg.dst)
    else:
        logger.error(f"{arg.src} is neither a File nor a Folder")


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
