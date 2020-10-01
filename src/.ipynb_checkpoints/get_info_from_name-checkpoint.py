# Built-In libraries
import os

# External libraries
import pandas as pd


def get_info_from_name(x):
    """

    :param x: str path
    :return: series
    """
    clean = os.path.splitext(os.path.basename(x))[0]
    clean = clean.split("_")
    clean[1] = "left" if clean[1] == "21015" else "right"
    return pd.Series(clean, index=["patient", "side", "samples", "replicates"])
