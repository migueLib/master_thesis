# Imports
import os
from os.path import isdir, isfile, join

import random
from glob import glob
import pandas as pd
from PIL import Image
from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset



class FundusDatasetRAM(Dataset):
    def __init__(self, imgs_dir=None, target_file=None, target=None, target_size=300, augment=True):
        # Size to re-scale
        self.target_size = target_size
        self.target = target
        
        # Image directory
        if imgs_dir is None:
            raise ValueError("No image directory provided")
        else:
            if isdir(imgs_dir):
                self.imgs_dir = imgs_dir
            else:
                raise NotADirectoryError(f"{imgs_dir} does not exist.")
    
        # Target file
        if target_file is None:
            raise ValueError("No target file provided")
        else:
            if isfile(target_file):
                self.target_file = target_file
            else:
                raise FileNotFoundError(f"{target_file} does not exist")
                
        # Load csv target file
        self.target_file = pd.read_csv(self.target_file)

        # Get classes and class dictionary
        self.classes = sorted(self.target_file[self.target].unique())
        self.dic = {k:v for v,k in enumerate(self.classes)}
        
        # Get id's from target file
        self.ids = self.target_file['file'].values
        
        # Load all images to RAM
        self.target_file["im"] = self.target_file["file"].apply(lambda x: io.imread(join(self.imgs_dir, x)))
        
        # Get augmentations
        self.augment = augment
        
        # Set augmentations to use on training 
        self.augment_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((300,300)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(90),
            transforms.ToTensor()
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((300,300)),
            transforms.ToTensor()
        ])

    def preprocess(self, img):
        # Resize image to desire size (for the network)
        # The data is so dirty we can't do much without cleaning it first ...
        
        # Augment the data
        if self.augment:
            img = self.augment_transform(img)
        else:
            img = self.test_transform(img)
            
        return img
    
    # 'cause size matters
    def __len__(self):
        return len(self.ids)

    # Returns iterable
    def __getitem__(self, i):
        # Get target
        target = self.target_file[self.target].iloc[i]
        
        # Get image
        img = self.target_file["im"].iloc[i]
        img = self.preprocess(img)

        return img, self.dic[target]
