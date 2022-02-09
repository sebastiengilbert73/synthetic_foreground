import logging
import argparse
import ast
import os
import cv2
import synthetic_foreground.foreground_generator as foreground_generator
import math

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main(
        foregroundImagesDirectory,
        imageSizeHW,
        outputDirectory
):
    logging.info("generate_pair.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    generator = foreground_generator.ForegroundGenerator(
        foreground_images_directory=foregroundImagesDirectory
    )

    (input_image, heatmap) = generator.Generate(
        image_sizeHW=imageSizeHW,
        maximum_number_of_trials=None,
        debug_directory=outputDirectory,
        background_image=None
    )

    cv2.imwrite(os.path.join(outputDirectory, "generatePair_main_input.png"), input_image)
    cv2.imwrite(os.path.join(outputDirectory, "generatePair_main_heatmap.png"), heatmap)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('foregroundImagesDirectory', help="The directory containing the foreground images")
    parser.add_argument('--imageSizeHW', help="The size of the generated images (Height, Width). Default: '(256, 256)'", default='(256, 256)')
    parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs_generate_pairs'",
                        default='./outputs_generate_pairs')
    args = parser.parse_args()

    imageSizeHW = ast.literal_eval(args.imageSizeHW)

    main(
        args.foregroundImagesDirectory,
        imageSizeHW,
        args.outputDirectory
    )