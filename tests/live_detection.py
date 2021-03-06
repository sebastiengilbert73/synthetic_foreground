import logging
import argparse
import os
import cv2
import ast
import architectures.heatmaps as arch
import torch
import numpy as np
import timeit
from timeit import default_timer as timer

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main(
        neuralNetworkFilepath,
        outputDirectory,
        image_sizeHW,
        cameraID,
        preprocessing
):
    logging.info("live_detection.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Load the neural network
    neural_network = None
    if os.path.basename(args.neuralNetworkFilepath).startswith("Assinica_16_32_64_"):
        neural_network = arch.Assinica(number_of_channels=(16, 32, 64), dropout_ratio=0.5)
    elif os.path.basename(args.neuralNetworkFilepath).startswith('Cascapedia_16_32_64'):
        neural_network = arch.Cascapedia(number_of_channels=(16, 32, 64), dropout_ratio=0.5)
    elif os.path.basename(args.neuralNetworkFilepath).startswith('Daaquam_32_64'):
        neural_network = arch.Daaquam(number_of_channels=(32, 64), dropout_ratio=0.5)
    elif os.path.basename(args.neuralNetworkFilepath).startswith('Resnet50'):
        neural_network = arch.Resnet50()
    else:
        raise NotImplementedError("live_detection.main(): Could not identify the architecture of '{}'".format(neuralNetworkFilepath))
    neural_network.load_state_dict(torch.load(neuralNetworkFilepath, map_location=torch.device('cpu')))
    neural_network.eval()

    capture = cv2.VideoCapture(cameraID)
    number_of_captures = 0
    captures_period = 50
    start = timer()
    while True:
        ret_val, image = capture.read()
        if ret_val == True:

            if image_sizeHW is not None:
                image = cv2.resize(image, (image_sizeHW[1], image_sizeHW[0]))

            input_tsr = None
            if preprocessing is not None:
                if preprocessing == 'blur3x3':
                    image = cv2.blur(image, (3, 3))
                else:
                    raise NotImplementedError("live_detection.main(): Not implemented preprocessing '{}'".format(preprocessing))
            # image.shape = (H, W, C)
            if input_tsr is None:
                input_img = np.moveaxis(image, 2, 0)  # (H, W, C) -> (C, H, W)
                input_tsr = (torch.from_numpy(input_img) / 256).unsqueeze(0)  # (1, C, H, W)

            # Pass through the neural network
            output_tsr = neural_network(input_tsr)
            heatmap = (256 * output_tsr).squeeze().detach().numpy().astype(np.uint8)

            # Display the image
            cv2.imshow('image', image)
            cv2.imshow('heatmap', heatmap)

            number_of_captures += 1
            if number_of_captures == captures_period:
                end = timer()
                delay_in_seconds = end - start
                fps = number_of_captures/delay_in_seconds
                logging.info("rate = {} fps".format(fps))
                start = timer()
                number_of_captures = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('neuralNetworkFilepath', help="The filepath to the neural network")
    parser.add_argument('--outputDirectory',
                        help="The output directory. Default: './outputs_live_detection'",
                        default='./outputs_live_detection')
    parser.add_argument('--imageSizeHW', help="The image resize (Height, Width), if desired. Default: 'None'",
                        default='None')
    parser.add_argument('--cameraID', help="The camera ID. Default: 0", type=int, default=0)
    parser.add_argument('--preprocessing', help="The preprocessing. Default: 'None'", default='None')
    args = parser.parse_args()

    image_sizeHW = None
    if args.imageSizeHW.upper() != 'NONE':
        image_sizeHW = ast.literal_eval(args.imageSizeHW)
    preprocessing = None
    if args.preprocessing.upper() != 'NONE':
        preprocessing = args.preprocessing

    main(
        args.neuralNetworkFilepath,
        args.outputDirectory,
        image_sizeHW,
        args.cameraID,
        preprocessing
    )