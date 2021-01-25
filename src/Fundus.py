# Built-in libraries
import os
import itertools

# External libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Fundus():

    def __init__(self, source=False, **kwargs):

        # Constructors depending on source
        if isinstance(source, str):
            try:
                self.im = self._image_from_file(source)
            except:
                print(source)


        # Get palette r, g, b
        self.r, self.g, self.b = self._get_rgb()


    @staticmethod
    def _image_from_file(path):
        return cv2.imread(path)

    def show(self):
        plt.matshow(self.im[:,:,::-1])
        plt.axis("off")

    def _get_rgb(self):
        r, g, b = self.im.T
        r, g, b = r.flatten(), g.flatten(), b.flatten()
        return r, g, b

    def get_radius(self, threshold = 10):
        """
        Gets the radius of a Fundus photograph at the x axis
        """
        # Gets the pixels in the midle (vertically) of the Fundus
        # Then sums their RGB values
        x = self.im[self.im.shape[0]//2, : , :].sum(1)

        # Use 1/10 of the mean as a threshold,
        # anything above count it as part of the eye
        # This value is the radius
        r = (x > x.mean()/threshold).sum()/2

        return r

    def get_rgb_mean(self):
        """
        Gets the mean values for RGB
        """
        return np.mean(self.r), np.mean(self.g), np.mean(self.b)

    def get_palette(self):
        return set(zip(self.r, self.g, self.b))

    def r_g(self):
        return len(set(zip(self.r, self.g)))

    def r_b(self):
        return len(set(zip(self.r, self.b)))

    def g_b(self):
        return len(set(zip(self.g, self.b)))

    def palette_len(self):
        return len(self.palette)
