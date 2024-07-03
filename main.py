# CSC320 Fall 2022
# Tutorial 1
# (c) Kyros Kutulakos, Towaki Takikawa, Esther Lin
#
# UPLOADING THIS CODE TO GITHUB OR OTHER CODE-SHARING SITES IS
# STRICTLY FORBIDDEN.
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY FORBIDDEN. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY.
#
# THE ABOVE STATEMENTS MUST ACCOMPANY ALL VERSIONS OF THIS CODE,
# WHETHER ORIGINAL OR MODIFIED.

import cv2
import glob
import numpy as np

def read_im(filename, colour):
    """Read in a .jpg or .png image

    Args:
        filename (str): path to image
        colour (int): imread flag for differentiating greyscale vs colour
        cv::IMREAD_GRAYSCALE = 0,
        cv::IMREAD_COLOR = 1, 

    Returns:
        (np.ndarray): image of shape
    """
    # TODO: replace with your implementation

    img = cv2.imread(filename, colour)
    return img


def write_im(filename, image):
    """Write a .jpg or .png image

    Args:
        filename (str): path to write to
        image (np.ndarray): image to be written
    """
    # TODO: replace with your implementation
    cv2.imwrite(filename, image)

"""
The general steps for creating a coloured misty water effect image are as follows:

    1. Read in all images in burst. We need a lot of images (20+) to create a smooth effect. 
    We assume that each image in burst has the same number of pixels and dimensions.

    2. Stack all images into a single array such that the stack has dimensions (K, N, M, 3). 
    Take a look at these NumPy docs for hints on doing the stacking. Here, K is the number of images 
    in our burst, N is the number of rows of pixels in each image, M is the number of colours of pixels. 
    3 represents our three colour channels RGB.

    3. Average over dimension 0 (K) to get the misty image of size (N,M,3). The result will be the misty effect image.
"""
def read_burst(dir, filetype, colour):
    """Read in all of the .jpg or .png images in a directory

    Args:
        filename (str): path to directory
        filetype (str): '.jpg' or '.png'
        colour (int): imread flag for differentiating greyscale vs colour

    Returns:
        (np.ndarray): image of shape (K, N, M, ...) where K is the number of images in dir
    """
    # TODO: replace with your implementation
    img_list = []
    for img in glob.iglob(f'{dir}/*'):
        if img.endswith(filetype):
            read_img = read_im(img, colour)
            img_list.append(read_img)
    
    return np.array(img_list)

def calculate_misty(image_stack):
    """Calculates the misty effect image

    Args:
        image_stack (np.ndarray): stack of images to be mistified, shape is (K, N, M, ...), where ... is the number of colour channels

    """
    # TODO: replace with your implementation
    return np.mean(image_stack, axis=0)

if __name__ == '__main__':

    # Part A.1
    # Load in an example image in colour
    img = read_im('./data/0001.jpg', cv2.IMREAD_COLOR)
    print("Colour", type(img), img.shape)

    # Write the same colour image, just as practice
    write_im('./results/original-colour.png',img)

    # Part A.2
    # Read in all images in the data directory in colour
    imgs = read_burst('./data', '.jpg', cv2.IMREAD_COLOR)

    # Calculate misty image
    misty_img = calculate_misty(imgs)

    # Save misty image
    write_im('./results/misty-colour.png', misty_img)

    # TODO: Part B, just a copy of part A but with different cv2 flag
    img = read_im('./data/0001.jpg', cv2.IMREAD_GRAYSCALE)
    print("Gray", type(img), img.shape)
    write_im('./results/original-grey.png',img)
    imgs = read_burst('./data', '.jpg', cv2.IMREAD_GRAYSCALE)
    misty_img = calculate_misty(imgs)
    write_im('./results/misty-grey.png', misty_img)
