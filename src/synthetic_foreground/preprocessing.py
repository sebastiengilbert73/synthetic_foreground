import numpy as np
import cv2


def DifferenceOfMedians(color_img, large_neighborhood_size, small_neighborhood_size):
    large_median_blurred_img = cv2.medianBlur(color_img, large_neighborhood_size)
    small_median_blurred_img = cv2.medianBlur(color_img, small_neighborhood_size)
    difference_img = np.clip((large_median_blurred_img.astype(np.int32) - small_median_blurred_img.astype(np.int32)), 0, 255).astype(np.uint8)
    return difference_img
