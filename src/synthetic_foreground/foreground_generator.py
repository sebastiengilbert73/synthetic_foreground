import abc
import random
import math
import numpy as np
import cv2
import logging
import os
import synthetic_heatmap.generator
import imutils

class ForegroundGenerator(synthetic_heatmap.generator.Generator):
    def __init__(self,
                 foreground_images_directory,
                 scale_range=[0.1, 0.7],  # The ratio of the foreground image with respect to the background
                 rotation_range=[0, 2 * math.pi],
                 hue_delta_range=[-30, 30],
                 foreground_luminance_inverse_threshold=220):
        parameters_dict = {'scale_range': scale_range,
                           'rotation_range': rotation_range,
                           'hue_delta_range': hue_delta_range}
        super().__init__(parameters_dict)

        self.foreground_images_directory = foreground_images_directory
        image_extensions = ['.JPG', '.JPEG', '.PNG', '.TIF']
        self.foreground_image_filepaths = [os.path.join(self.foreground_images_directory, f) for f in
                         os.listdir(self.foreground_images_directory)
                         if any(f.upper().endswith(ext) for ext in image_extensions)]
        self.foreground_luminance_inverse_threshold = foreground_luminance_inverse_threshold

    def Generate(self,
                 image_sizeHW,
                 maximum_number_of_trials=None,
                 debug_directory=None,
                 background_image=None
                 ):
        heatmap = np.zeros(image_sizeHW, dtype=np.uint8)
        # = np.ones((3, 3), dtype=np.uint8)

        # Create the background image
        input_image = background_image
        result_msg = None
        if background_image is None:
            input_image, result_msg = synthetic_heatmap.generator.DownloadRandomImage(image_sizeHW=image_sizeHW)
        if input_image.shape != image_sizeHW:
            input_image = cv2.resize(input_image, image_sizeHW)


        # Select a foreground image
        foreground_filepath = random.choice(self.foreground_image_filepaths)
        foreground_img = cv2.imread(foreground_filepath, cv2.IMREAD_COLOR)
        if debug_directory is not None:
            cv2.imwrite(os.path.join(debug_directory, "ForegroundGenerator_Generate_background.png"), input_image)
            cv2.imwrite(os.path.join(debug_directory, "ForegroundGenerator_Generate_foreground.png"), foreground_img)

        # Transform the foreground image
        # Scaling
        scale = self.RandomValueInRange('scale_range')
        foreground_sizeHW = (round(scale * input_image.shape[0]), round(scale * input_image.shape[1]))
        foreground_img = cv2.resize(foreground_img, (foreground_sizeHW[1], foreground_sizeHW[0]))
        # Rotation
        rotation_theta = self.RandomValueInRange('rotation_range')
        foreground_img = imutils.rotate(foreground_img, 180/math.pi * rotation_theta)
        # Mask
        foreground_hls = cv2.cvtColor(foreground_img, cv2.COLOR_BGR2HLS)
        _, foreground_mask = cv2.threshold(foreground_hls[:, :, 1], 0, 255, cv2.THRESH_BINARY)
        _, foreground_below_threshold = cv2.threshold(foreground_hls[:, :, 1], self.foreground_luminance_inverse_threshold, 255, cv2.THRESH_BINARY_INV)
        foreground_mask = cv2.min(foreground_mask, foreground_below_threshold)
        kernel_3x3 = np.ones((3, 3), dtype=np.uint8)
        foreground_mask = cv2.erode(foreground_mask, kernel_3x3)
        foreground_mask = cv2.dilate(foreground_mask, kernel_3x3)
        # Hue change
        hue_delta = self.RandomValueInRange('hue_delta_range', must_be_rounded=True) % 180
        foreground_hls[:, :, 0] += hue_delta
        foreground_img = cv2.cvtColor(foreground_hls, cv2.COLOR_HLS2BGR)

        if debug_directory is not None:
            cv2.imwrite(os.path.join(debug_directory, "ForegroundGenerator_Generate_foregroundMask.png"), foreground_mask)

        input_image[0: foreground_img.shape[0], 0: foreground_img.shape[1], :] = foreground_img

        return (input_image, heatmap)