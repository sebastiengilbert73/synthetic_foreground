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
                 scale_range=[0.2, 0.7],  # The ratio of the foreground image with respect to the background
                 rotation_range=[0, 2 * math.pi],
                 hue_delta_range=[-15, 15],
                 luminance_delta_range=[-15, 15],
                 foreground_luminance_inverse_threshold=220):
        parameters_dict = {'scale_range': scale_range,
                           'rotation_range': rotation_range,
                           'hue_delta_range': hue_delta_range,
                           'luminance_delta_range': luminance_delta_range}
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
                 background_image=None,
                 add_uniform_square=False,
                 number_of_foreground_objects=1
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

        # Add a square of uniform color: the goal is to prevent the neural network to rely on a specific color
        if add_uniform_square:
            square_scale = self.RandomValueInRange('scale_range')
            square_size = round(square_scale * input_image.shape[0])
            square_anchor_pt = (int((image_sizeHW[1] - square_size) * random.random()),
                         int((image_sizeHW[0] - square_size) * random.random()))
            square_color = np.random.randint(0, 256, size=3).tolist()
            cv2.rectangle(input_image, square_anchor_pt, (square_anchor_pt[0] + square_size, square_anchor_pt[1] + square_size), square_color, thickness=-1)

        if debug_directory is not None:
            cv2.imwrite(os.path.join(debug_directory, "ForegroundGenerator_Generate_background.png"), input_image)


        for objectNdx in range(number_of_foreground_objects):
            # Select a foreground image
            foreground_filepath = random.choice(self.foreground_image_filepaths)
            foreground_img = cv2.imread(foreground_filepath, cv2.IMREAD_COLOR)
            # Flip half of the time
            if random.random() > 0.5:
                foreground_img = cv2.flip(foreground_img, flipCode=1)

            if debug_directory is not None:
                cv2.imwrite(os.path.join(debug_directory, "ForegroundGenerator_Generate_foreground{}.png".format(objectNdx)), foreground_img)

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
            foreground_hls = foreground_hls.astype(np.int32)
            # Hue change
            hue_delta = self.RandomValueInRange('hue_delta_range', must_be_rounded=True)
            foreground_hls[:, :, 0] += hue_delta
            foreground_hls[:, :, 0] = foreground_hls[:, :, 0] % 180

            # Luminance change
            luminance_delta = self.RandomValueInRange('luminance_delta_range', must_be_rounded=True)
            foreground_hls[:, :, 1] += luminance_delta
            foreground_hls[:, :, 1] = foreground_hls[:, :, 1] % 256

            foreground_hls = np.clip(foreground_hls, 0, 255).astype(np.uint8)
            foreground_img = cv2.cvtColor(foreground_hls, cv2.COLOR_HLS2BGR)

            # Choose an anchor point
            anchor_pt = (int( (image_sizeHW[1] - foreground_img.shape[1]) * random.random()), int((image_sizeHW[0] - foreground_img.shape[0]) * random.random()))
            # Draw the foreground
            for y in range(anchor_pt[1], anchor_pt[1] + foreground_img.shape[0]):
                for x in range(anchor_pt[0], anchor_pt[0] + foreground_img.shape[1]):
                    foreground_pt = (x - anchor_pt[0], y - anchor_pt[1])
                    if foreground_mask[foreground_pt[1], foreground_pt[0]] > 0:
                        input_image[y, x, :] = foreground_img[foreground_pt[1], foreground_pt[0]]
                        heatmap[y, x] = 255
        # Remove a row of pixels in the heatmap
        heatmap = cv2.erode(heatmap, kernel_3x3)

        # Blur the heatmap periphery
        kernel_5x5 = np.ones((5, 5), dtype=np.uint8)
        dilated_heatmap = cv2.dilate(heatmap, kernel_5x5)
        eroded_heatmap = cv2.erode(heatmap, kernel_3x3)
        heatmap_periphery = dilated_heatmap - eroded_heatmap
        blurred_img = cv2.blur(input_image, (3, 3))
        for y in range(input_image.shape[0]):
            for x in range(input_image.shape[1]):
                if heatmap_periphery[y, x] > 0:
                    input_image[y, x, :] = blurred_img[y, x]


        if debug_directory is not None:
            cv2.imwrite(os.path.join(debug_directory, "ForegroundGenerator_Generate_foregroundMask.png"), foreground_mask)
            cv2.imwrite(os.path.join(debug_directory, "ForegroundGenerator_Generate_heatmapPeriphery.png"), heatmap_periphery)
        #input_image[0: foreground_img.shape[0], 0: foreground_img.shape[1], :] = foreground_img

        return (input_image, heatmap)