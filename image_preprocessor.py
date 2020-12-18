import cv2
import numpy as np
import datetime
import tensorflow as tf

def create_circular_mask_1(image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    cropped_img_array = tf.image.central_crop(img_array, 0.85)
    cropped_img = tf.keras.preprocessing.image.array_to_img(cropped_img_array)
    return cropped_img


def hair_removal(image: str, inpaint_radius: int = 40):
    """ Removes noise(hair) in the input image
    Args:
        image (str): path to image
        inpaint_radius (int): radius of a circular neighborhood of each point inpainted
    Returns:
        restored_image: restores the image with lesser noise using the region neighborhood
    """
    # Read image and convert to gray scale
    im = cv2.imread(image)
    grayScale = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    # Get kernel for Morphological Transformation
    kernel = cv2.getStructuringElement(1, (17, 17))
    # Perform black-hat transformation
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    # Create a Binary image of the hair and paint over with `inpaint_radius` px
    # neighouring cells
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(im, thresh2, inpaintRadius=inpaint_radius,
                      flags=cv2.INPAINT_TELEA)
    return dst


def create_circular_mask(img, radius: int = 350):
    """ Creates a circular mask
    Args:
        image (str): path to image
    """
    # Read the image
    height, width, depth = img.shape
    # Create circle mask for the image
    circle_img = np.zeros((int(height), int(width)), np.uint8)
    cv2.circle(circle_img, (int(width/2), int(height/2)),
               radius=radius, color=(255, 0, 0), thickness=-1)
    # Create circular mask Bit-wise conjunction
    masked_data = cv2.bitwise_and(img, img, mask=circle_img)
    return masked_data


def change_contrast(img, cliplimit: float):
    """Performs Contrast-limited Adaptive Histogram Equalization (CLAHE)
    Args:
        image (str): path to image
        cliplimit (float): parameter sets the threshold for contrast
        limiting, defaults to 40. 
    Returns:
        image after CLAHE
    """
    # Conversion of RGB to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Apply only to Lightness component
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    # Merge back the image
    lab = cv2.merge(lab_planes)
    # Convert back to RGB
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


def resize_image(image: str, width: int, height: int):
    """Resize images to defined `width` x `height`
    Args:
        image(str): path to image 
        width (int): width of resized image
        height (int): height of resized image
    """
    im = cv2.imread(image)
    # Resize image
    resized_image = cv2.resize(im, dsize=(width, height))
    return resized_image


def k_means_segementation(im, attempts: int):
    """Performs segementation of image according to k-means clustering
    Args:
        image (str): path to image
        image (int):  number of times the algorithm is executed using different initial labellings
    Returns:
        Segmented image
    """
    # Convert to gray image

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Reshape and convert to float32 data
    Z = gray.reshape((-1, 1))
    Z = np.float32(Z)
    # Perform k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(
        Z, 2, None, criteria, attempts=attempts, flags=cv2.KMEANS_RANDOM_CENTERS)
    # Get mask matrices, White--> label"1", Black--> label"0"
    A = Z[label.ravel() == 0]
    B = Z[label.ravel() == 1]
    # Convert data into 8-bit values
    center = np.uint8(center)
    
    segmented_data = center[label.flatten()]
    
    # Reshape to original image dimensions
    segmented_image = segmented_data.reshape((gray.shape))
    

    return segmented_image




