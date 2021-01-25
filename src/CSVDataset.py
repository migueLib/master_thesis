import os
import torch
import pandas as pd
from skimage import io
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    """UK Biobank Fundus dataset."""

    def __init__(self, csv_file, root_dir, target, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target = target
        self.classes = list(set(self.frame[self.target]))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Getting full image path
        img_name = os.path.join(self.root_dir, self.frame["name"].iloc[idx])

        # Read image
        image = io.imread(img_name)

        # Read class for the image
        im_class = self.frame[self.target].iloc[idx]

        if self.transform:
            image = self.transform(image)

        return image, im_class
