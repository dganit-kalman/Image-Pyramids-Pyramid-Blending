import traceback

import sys
from imageio import imread
import numpy as np
from skimage.color import rgb2gray
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import os

MAX_GRAY_LEVEL = 255


def read_image(filename, representation):
    """
    reads an image file and converts it into a given representation.
    :param file_name: the filename of an image on disk
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
        image (1) or an RGB image (2).
    :return: an image based on the desired representation.
    """
    picture = imread(filename).astype(np.float64)
    if representation == 1:
        return rgb2gray(picture)
    elif representation == 2:
        picture /= MAX_GRAY_LEVEL #normalized the picture.
        return picture
    else:
        print("Entered wrong number to representation")


def build_ker(kernel_size):
    """
    builds the 2D gaussian kernel
    :param kernel_size: is the size of the gaussian kernel in each dimension (an odd integer).
    :return: 2D gaussian kernel with size kernel_size X kernel_size
    """
    base_to_ker = np.array([1, 1])
    row_gaus_ker = np.array([1, 1])
    for i in range(kernel_size - 2):  # size-2 because each iter raise the length by 1 and we start with length 2
        row_gaus_ker = np.convolve(row_gaus_ker, base_to_ker)
    col_gaus_ker = row_gaus_ker.reshape([1, kernel_size])
    return col_gaus_ker


def reduce(im, filter_size):
    """
    reduce the size of the picture and blur the image
    :param im: a grayscale image
    :param filter_size: the size of the Gaussian filter
    :return: the reduced image
    """
    filter_vec = build_ker(filter_size)
    gaus_ker = scipy.signal.convolve2d(filter_vec, filter_vec.reshape([filter_size, 1])).astype(np.float64)
    gaus_ker = gaus_ker / np.sum(gaus_ker)  # divide by sum because the kernel elements should sum to 1
    blur = scipy.ndimage.filters.convolve(im.astype(np.float64), gaus_ker).astype(np.float64)
    sub_sample = blur[::2, ::2]
    return sub_sample


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    construct a Gaussian pyramid of a given image
    :param im: a grayscale image
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter
    :return: the gaussian pyramid and s row vector used for the pyramid construction
    """
    filter_vec = build_ker(filter_size)
    pyr = [im]
    gaus_lev_pyr = reduce(im, filter_size)
    for i in range(max_levels - 1):
        if gaus_lev_pyr.shape[0] < 16 or gaus_lev_pyr.shape[1] < 16:
            break
        pyr.append(gaus_lev_pyr.astype(np.float64))
        gaus_lev_pyr = reduce(gaus_lev_pyr, filter_size)
    return pyr, filter_vec


def expand(im, filter_vec):
    """
    expands the image by zero padding and blur the image
    :param filter_vec: the filter vector we used for convolve with the image
    :param im: a grayscale image
    :return: the expanded image
    """
    zero_padding = np.zeros([im.shape[0] * 2, im.shape[1] * 2])
    zero_padding[::2, ::2] = im
    lap_ker = scipy.signal.convolve2d(filter_vec, filter_vec.reshape([filter_vec.shape[1], 1])).astype(np.float64)
    blur = scipy.ndimage.filters.convolve(zero_padding, lap_ker).astype(np.float64)
    return blur


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    construct a Laplacian pyramid of a given image
    :param im: a grayscale image
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter
    :return: the laplacian pyramid and s row vector used for the pyramid construction
    """
    filter_vec = build_ker(filter_size)
    filter_vec1 = filter_vec*2/np.sum(filter_vec)
    gaus_pyr = build_gaussian_pyramid(im, max_levels, filter_size)[0]
    pyr = []
    for i in range(max_levels - 1):
        lap_lev_pyr = gaus_pyr[i] - expand(gaus_pyr[i + 1], filter_vec1)
        pyr.append(lap_lev_pyr.astype(np.float64))
    pyr.append(gaus_pyr[max_levels - 1])
    return pyr, filter_vec1


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    reconstruct an image from its Laplacian Pyramid
    :param lpyr: the Laplacian pyramid
    :param filter_vec: the filter that are generated the image
    :param coeff: every level at the pyramid we should multiply by its corresponding coefficient coeff[i].
    :return: the reconstructed image
    """
    for i in range(len(lpyr)):
        lpyr[i] *= coeff[i]
    im = lpyr[len(lpyr) - 1]
    for i in range(len(lpyr) - 2, -1, -1):
        im = expand(im, filter_vec) + lpyr[i]
    return im.astype(np.float64)


def render_pyramid(pyr, levels):
    """
    helps for displaying the desired image
    :param pyr: is either a Gaussian or Laplacian pyramid
    :param levels: the number of levels to present in the result ≤ max_levels
    :return: a single black image in which the pyramid levels of the given pyramid pyr are stacked horizontally
    """
    for i in range(len(pyr)):
        pyr[i] = (pyr[i] - np.amin(pyr[i]))
        pyr[i] = pyr[i] / np.amax(pyr[i])
    row = pyr[0].shape[0]
    col = 0
    for i in range(levels):
        col += pyr[i].shape[1]
    res = np.zeros([row, col])
    im_place_col = 0
    for i in range(levels):
        pyr_height, pyr_width = pyr[i].shape
        res[:pyr_height, im_place_col: im_place_col + pyr_width] = pyr[i]
        im_place_col += pyr_width
    return res


def display_pyramid(pyr, levels):
    """
    display the stacked pyramid image
    :param pyr: is either a Gaussian or Laplacian pyramid
    :param levels: the number of levels to present in the result ≤ max_levels
    """
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap='gray')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Implement pyramid blending
    :param im1: input grayscale image to be blended.
    :param im2: input grayscale images to be blended.
    :param mask: is a boolean mask representing which parts of im1 and im2 should appear in the resulting im_blend.
    :param max_levels: the max_levels parameter when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im: the size of the Gaussian filter which defining the filter used in the construction of the
        Laplacian pyramids of im1 and im2.
    :param filter_size_mask: the size of the Gaussian filter(an odd scalar that represents a squared filter) which
        defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: a blending image
    """
    L1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    Gm = build_gaussian_pyramid(mask.astype(np.double), max_levels, filter_size_mask)[0]
    Lout = []
    for i in range(max_levels):
        Lout.append(np.multiply(Gm[i],L1[i]) + np.multiply((1 - Gm[i]),(L2[i])))
    im_blend = laplacian_to_image(Lout, filter_vec, np.ones(max_levels))
    return np.clip(im_blend, 0, 1)


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)

def blending_example1():
    im1 = read_image(relpath("external/netta.jpg"), 2)
    im2 = read_image(relpath("external/rothLor.jpg"), 2)
    mask = read_image(relpath("external/mask3.jpg"), 1)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    im_blend = np.zeros([im1.shape[0], im1.shape[1],3])
    for i in range(3):
        im_blend[...,i] = pyramid_blending(im1[...,i], im2[...,i], mask, 5, 3, 3)
    plt.figure()
    plt.subplot(221)
    plt.imshow(im1)
    plt.subplot(222)
    plt.imshow(im2)
    plt.subplot(223)
    plt.imshow(mask, cmap='gray')
    plt.subplot(224)
    plt.imshow(im_blend)
    plt.show()
    return im1, im2, mask.astype(np.bool), im_blend


def blending_example2():
    im1 = read_image(relpath("external/VanGoghbed.jpg"), 2)
    im2 = read_image(relpath("external/think2.jpg"), 2)
    mask = read_image(relpath("external/maskthink2.jpg"), 1)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    im_blend = np.zeros([im1.shape[0], im1.shape[1], 3])
    for i in range(3):
        im_blend[..., i] = pyramid_blending(im1[..., i], im2[..., i], mask, 3, 10, 3)
    plt.figure()
    plt.subplot(221)
    plt.imshow(im1)
    plt.subplot(222)
    plt.imshow(im2)
    plt.subplot(223)
    plt.imshow(mask, cmap='gray')
    plt.subplot(224)
    plt.imshow(im_blend)
    plt.show()
    return im1, im2, mask.astype(np.bool), im_blend
blending_example1()