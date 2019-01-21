from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from scipy import ndimage, misc
from skimage import data
import matplotlib.pyplot as plt
import six.moves as sm
import re
import os
from collections import defaultdict
import PIL.Image
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

np.random.seed(44)
ia.seed(44)

def main():
    for i in range(1, 181):
        toAdd = ''
        if i < 10:
            toAdd = 'no-hits000' + str(i)
            print(toAdd)
        if i >= 10 and i < 100:
            toAdd = 'no-hits00' + str(i)
        if i >= 100:
            toAdd = 'no-hits0' + str(i)
        draw_single_sequential_images(toAdd, "no-hits", "no-hits-aug")
    for i in range(1, 211):
        toAdd = ''
        if i < 10:
            toAdd = 'straights000' + str(i)
        if i >= 10 and i < 100:
            toAdd = 'straights00' + str(i)
        if i >= 100:
            toAdd = 'straights0' + str(i)
        draw_single_sequential_images(toAdd, "straights", "straights-aug")
    for i in range(1, 191):
        toAdd = ''
        if i < 10:
            toAdd = 'uppercuts000' + str(i)
        if i >= 10 and i < 100:
            toAdd = 'uppercuts00' + str(i)
        if i >= 100:
            toAdd = 'uppercuts0' + str(i)
        draw_single_sequential_images(toAdd, "uppercuts", "uppercuts-aug")

def draw_single_sequential_images(filename, path, aug_path):
    ia.seed(44)

    image = misc.imresize(ndimage.imread(path + "/" + filename + ".jpg"), (56, 100))

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-5, 5),
                shear=(-5, 5),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            )),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            iaa.Invert(0.05, per_channel=False),
            iaa.SomeOf((0, 5),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 2.0)),
                        iaa.AverageBlur(k=(2, 5)),
                        iaa.MedianBlur(k=(3, 5)),
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5),
                    iaa.Add((-10, 10), per_channel=0.5),
                    iaa.AddToHueAndSaturation((-20, 20)),
                    iaa.OneOf([
                        iaa.Multiply((0.9, 1.1), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-2, 0),
                            first=iaa.Multiply((0.9, 1.1), per_channel=True),
                            second=iaa.ContrastNormalization((0.9, 1.1))
                        )
                    ]),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                ],
                random_order=True
            )
        ],
        random_order=True
    )

    im = np.zeros((1, 56, 100, 3), dtype=np.uint8)
    for c in range(0, 1):
        im[c] = image

    grid = seq.augment_images(im)
    for im in range(len(grid)):
        misc.imsave(aug_path + "/" + filename + "_" + str(im) + ".jpg", grid[im])

if __name__ == "__main__":
    main()