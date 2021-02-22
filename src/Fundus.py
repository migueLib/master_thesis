# Built-in libraries
import os
import itertools

# External libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Fundus():
    
    # Constructors depending on source, if source is file use opencv
    def __init__(self, source=False, **kwargs):
        self.path = None
        
        # Check if file exists if not print error
        if os.path.isfile(source):
            self.im = self._image_from_file(source)
            self.path = source
        else:
            raise TypeError (f"{source} is not valid")
        
        # Get H, W, C
        self.h, self.w, self.c = self.im.shape
        
        # Get palette r, g, b
        self.r, self.g, self.b = self._get_rgb()

    @staticmethod
    def _image_from_file(path):
        return cv2.imread(path)

    def _get_rgb(self):
        r, g, b = self.im.T
        r, g, b = r.flatten(), g.flatten(), b.flatten()
        return r, g, b

    def show(self):
        plt.matshow(self.im[:, :, ::-1])
        plt.axis("off")
    
    def get_radius(self, threshold=1):
        """
        Gets the radius of a Fundus photograph at the x axis
        """
        # Gets the pixels in the middle (h//2) of the Fundus
        # Then sums their RGB values
        x = self.im[self.im.shape[0] // 2, :, :].sum(1)

        # Use a threshold, anything above it counts as part of the eye
        # This value is the radius
        r = (x >= threshold).sum() // 2

        return r
    
    def apply_mask(self, mask):
        """
        Applies any mask to the Fundus image
        """
        self.im = cv2.bitwise_and(self.im, self.im, mask=mask)
        
    def apply_circular_mask(self, r):
        """
        Applies a circular mask, based on a given radius
        to the Fundus image
        """
        mask = self.get_circular_mask(r)
        self.apply_mask(mask)
        
    def apply_center_crop(self, w, h):
        """
        Applies a center cropp, takes new width and height 
        """
        # Get middle point 
        xc = self.w//2
        yc = self.h//2

        # Get new y and x 
        x = xc - w//2
        y = yc - h//2

        self.im = self.im[int(y):int(y+h), int(x):int(x+w)]
        
    def apply_scale_radius(self, scale, threshold=1):
        """
        Resizes the image based on a radial scale
        """
        s = (scale*1.0)/self.get_radius(threshold)
        
        self.im = cv2.resize(self.im, (0,0), fx=s, fy=s)

    
    def get_circular_mask(self, r):
        """
        Calculates a centered circular mask
        """
        mask = np.zeros((self.im.shape[0], self.im.shape[1]), dtype=np.uint8)
        cv2.circle(img=mask, center=(self.im.shape[1] // 2, self.im.shape[0] // 2), radius=int(r),
                   color=(1, 1, 1, 1), thickness=-1, lineType=8, shift=0)
        return mask
    
    def get_threshold_mask(self, channel=2, threshold=10):
        """
        Generates a mask based on an RGB value
        """
        mask = (self.img[:,:,channel] > threshold).astype(np.uint8)
        return mask

    def normalize(self, r=None):
        """
        Normalizes Fundus image
        """
        # Calculate the radius to normalize
        r = r if r is not None else self.get_radius()
        
        # Calculate the Gaussian Blur based on the radius
        try:
            gaussian_blur = cv2.GaussianBlur(src=self.im, ksize=(0, 0), sigmaX=r / 30)
        except:
            raise ValueError(f'r={r} for {self.path}')

        # Blend GB and Original Image
        normalized_im = cv2.addWeighted(src1=self.im, alpha=4, src2=gaussian_blur, beta=-4, gamma=128)

        # Create a circular mask to remove the outer 5%
        # (This will avoid frontier effects)
        mask = self.get_circular_mask(r*0.95)

        # Apply mask
        normalized_im = cv2.bitwise_and(normalized_im, normalized_im, mask=mask)

        return normalized_im

    def get_rgb_mean(self):
        """
        Gets the mean values for RGB
        """
        return np.mean(self.r), np.mean(self.g), np.mean(self.b)
    
    def get_rgb_std(self):
        """
        Gets the standar deviation for the RGB values of the fundus
        """
        return np.std(self.r), np.std(self.g), np.std(self.b)
        

    def get_palette(self):
        return set(zip(self.r, self.g, self.b))

    def r_g(self):
        return len(set(zip(self.r, self.g)))

    def r_b(self):
        return len(set(zip(self.r, self.b)))

    def g_b(self):
        return len(set(zip(self.g, self.b)))
