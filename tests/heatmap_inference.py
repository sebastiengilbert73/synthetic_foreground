import logging
import argparse
import os
import torch
import architectures.heatmaps as arch
import cv2
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main(
        imageFilepath,
        neuralNetworkFilepath,
        outputDirectory,
        preprocessing
    ):
    logging.info("heatmap_inference.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    neural_network = None
    if os.path.basename(args.neuralNetworkFilepath).startswith("Assinica_16_32_64_"):
        neural_network = arch.Assinica(number_of_channels=(16, 32, 64), dropout_ratio=0.5)
    elif os.path.basename(args.neuralNetworkFilepath).startswith('Cascapedia_16_32_64'):
        neural_network = arch.Cascapedia(number_of_channels=(16, 32, 64), dropout_ratio=0.5)
    else:
        raise NotImplementedError("heatmap_inference.main(): Not implemented architecture '{}'".format(architecture))
    neural_network.load_state_dict(torch.load(neuralNetworkFilepath, map_location=torch.device('cpu')))
    neural_network.eval()

    input_img = cv2.imread(imageFilepath, cv2.IMREAD_COLOR)
    input_tsr = None
    if preprocessing is not None:
        if preprocessing == 'grayscale':
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img = np.expand_dims(input_img, axis=2)  # (H, W) -> (H, W, 1)
        else:
            raise NotImplementedError("heatmap_inference.main(): Not implemented preprocessing '{}'".format(preprocessing))
    if input_tsr is None:
        input_img = np.moveaxis(input_img, 2, 0)  # (H, W, C) -> (C, H, W)
        input_tsr = (torch.from_numpy(input_img)/256).unsqueeze(0)

    output_tsr = neural_network(input_tsr)
    output_img = (256 * output_tsr).squeeze().detach().numpy()
    cv2.imwrite(os.path.join(outputDirectory, "heatmap.png"), output_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imageFilepath', help="The filepath to the image")
    parser.add_argument('neuralNetworkFilepath', help="The filepath to the neural network")
    parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs_heatmap_inference'", default='./outputs_heatmap_inference')
    parser.add_argument('--preprocessing', help="The image preprocessing. Default: 'None'",
                        default='None')
    args = parser.parse_args()

    preprocessing = args.preprocessing
    if preprocessing.upper() == 'NONE':
        preprocessing = None

    main(
        args.imageFilepath,
        args.neuralNetworkFilepath,
        args.outputDirectory,
        preprocessing
    )