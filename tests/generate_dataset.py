import logging
import argparse
import os
import numpy as np
import synthetic_foreground.foreground_generator as foreground_generator
import cv2
import ast

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main(
        foregroundImagesDirectory,
        outputDirectory,
        numberOfPairs,
        imageSizeHW,
        scaleRange,
        rotationRange,
        hueDeltaRange,
        luminanceDeltaRange,
        foregroundLuminanceInverseThreshold,
        numberOfForegroundObjects
):
    logging.info("generate_dataset.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    generator = foreground_generator.ForegroundGenerator(
        foreground_images_directory=foregroundImagesDirectory,
        scale_range=scaleRange,
        rotation_range=rotationRange,
        hue_delta_range=hueDeltaRange,
        luminance_delta_range=luminanceDeltaRange,
        foreground_luminance_inverse_threshold=foregroundLuminanceInverseThreshold
    )

    for pairNdx in range(numberOfPairs):
        (input_img, heatmap) = generator.Generate(
            image_sizeHW=imageSizeHW,
            maximum_number_of_trials=None,
            debug_directory=None,
            background_image=None,
            number_of_foreground_objects=numberOfForegroundObjects
        )
        image_filepath = os.path.join(outputDirectory, "image_{}.png".format(pairNdx))
        heatmap_filepath = os.path.join(outputDirectory, "heatmap_{}.png".format(pairNdx))
        cv2.imwrite(image_filepath, input_img)
        cv2.imwrite(heatmap_filepath, heatmap)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('foregroundImagesDirectory', help="The directory containing the foreground images")
    parser.add_argument('--outputDirectory',
                        help="The output directory. Default: './outputs_generate_dataset'",
                        default='./outputs_generate_dataset')
    parser.add_argument('--numberOfPairs', help="The number of generated pairs (image and heatmap). Default: 256",
                        type=int, default=256)
    parser.add_argument('--imageSizeHW', help="The image size (Height, Width). Default: '(256, 256)'",
                        default='(256, 256)')
    parser.add_argument('--scaleRange', help="The scale of the foreground with respect to the background size. Default: '(0.2, 0.7)'", default='(0.2, 0.7)')
    parser.add_argument('--rotationRange', help="The rotation range for the foreground image. in radians. Default: '(0, 6.28)'", default='(0, 6.28)')
    parser.add_argument('--hueDeltaRange', help="The range of hue change for the foreground image. Default: '(-15, 15)'", default='(-15, 15)')
    parser.add_argument('--luminanceDeltaRange', help="The range of luminance change for the foreground image. Default: '(-15, 15)'", default='(-15, 15)')
    parser.add_argument('--foregroundLuminanceInverseThreshold', help="The luminance inverse threshold, to segment the foreground object. Default: 220", type=int, default=220)
    parser.add_argument('--numberOfForegroundObjects', help="The number of foreground objects per image. Default: 1", type=int, default=1)
    args = parser.parse_args()

    imageSizeHW = ast.literal_eval(args.imageSizeHW)
    scaleRange = ast.literal_eval(args.scaleRange)
    rotationRange = ast.literal_eval(args.rotationRange)
    hueDeltaRange = ast.literal_eval(args.hueDeltaRange)
    luminanceDeltaRange = ast.literal_eval(args.luminanceDeltaRange)

    main(
        args.foregroundImagesDirectory,
        args.outputDirectory,
        args.numberOfPairs,
        imageSizeHW,
        scaleRange,
        rotationRange,
        hueDeltaRange,
        luminanceDeltaRange,
        args.foregroundLuminanceInverseThreshold,
        args.numberOfForegroundObjects
    )