import logging
import argparse
import ast
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import random
import architectures.heatmaps as arch
import sys
import synthetic_foreground.foreground_generator as foreground_generator

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

class ImageHeatmapDataset(Dataset):
    def __init__(self, inputOutputfilepaths, image_sizeHW, preprocessing):
        self.inputOutputFilepaths = inputOutputfilepaths
        self.image_sizeHW = image_sizeHW
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.inputOutputFilepaths)

    def __getitem__(self, idx):
        (input_img_filepath, output_img_filepath) = self.inputOutputFilepaths[idx]

        input_img = cv2.imread(input_img_filepath, cv2.IMREAD_COLOR)  # (H, W, C)
        if input_img.shape != self.image_sizeHW:
            input_img = cv2.resize(input_img, self.image_sizeHW)
        if self.preprocessing is not None:
            if preprocessing == 'grayscale':
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)  # (W, W)
                input_img = np.expand_dims(input_img, axis=2)  # (H, W) -> (H, W, 1)
            else:
                raise NotImplementedError("ImageHeatmapDataset.__getitem__(): Not implemented preprocessing '{}'".format(self.preprocessing))

        input_img = np.moveaxis(input_img, 2, 0)  # (H, W, C) -> (C, H, W)
        output_img = cv2.imread(output_img_filepath, cv2.IMREAD_GRAYSCALE)
        if output_img.shape != self.image_sizeHW:
            output_img = cv2.resize(output_img, self.image_sizeHW)
        output_img = np.expand_dims(output_img, axis=0)  # (H, W) -> (1, H, W)

        input_tsr = torch.from_numpy(input_img)/256
        output_tsr = torch.from_numpy(output_img)/256

        return (input_tsr, output_tsr)

class GeneratedImageHeatmapDataset(Dataset):
    def __init__(self,
                 foreground_images_directory,
                 image_sizeHW,
                 dataset_size,
                 preprocessing,
                 scale_range,
                 rotation_range,
                 hue_delta_range,
                 luminance_delta_range,
                 foreground_luminance_inverse_threshold,
                 number_of_foreground_objects):
        self.generator = foreground_generator.ForegroundGenerator(
            foreground_images_directory=foreground_images_directory,
            scale_range=scale_range,  # The ratio of the foreground image with respect to the background
            rotation_range=rotation_range,
            hue_delta_range=hue_delta_range,
            luminance_delta_range=luminance_delta_range,
            foreground_luminance_inverse_threshold=foreground_luminance_inverse_threshold
        )
        self.image_sizeHW = image_sizeHW
        self.dataset_size = dataset_size  # Regulates the number of batches in an epoch
        self.preprocessing = preprocessing
        self.number_of_foreground_objects = number_of_foreground_objects

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        (input_img, heatmap) = self.generator.Generate(self.image_sizeHW, number_of_foreground_objects=self.number_of_foreground_objects)
        if self.preprocessing is not None:
            if preprocessing == 'grayscale':
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)  # (W, W)
                input_img = np.expand_dims(input_img, axis=2)  # (H, W) -> (H, W, 1)
            else:
                raise NotImplementedError("GeneratedImageHeatmapDataset.__getitem__(): Not implemented preprocessing '{}'".format(self.preprocessing))
        input_img = np.moveaxis(input_img, 2, 0)  # (H, W, C) -> (C, H, W)
        heatmap = np.expand_dims(heatmap, axis=0)  # (H, W) -> (1, H, W)

        input_tsr = torch.from_numpy(input_img) / 256
        output_tsr = torch.from_numpy(heatmap) / 256

        return (input_tsr, output_tsr)


def main(
        foregroundImagesDirectory,
        validationDirectory,
        outputDirectory,
        image_sizeHW,
        useCuda,
        heatmapKeyword,
        inputImageKeyword,
        batchSize,
        architecture,
        dropoutRatio,
        learningRate,
        momentum,
        weightDecay,
        numberOfEpochs,
        numberOfBatchesPerEpoch,
        startingNeuralNetworkFilepath,
        preprocessing,
        scaleRange,
        rotationRange,
        hueDeltaRange,
        luminanceDeltaRange,
        foregroundLuminanceInverseThreshold,
        numberOfForegroundObjects
):
    logging.info("train_heatmap.main() useCuda = {}".format(useCuda))
    device = 'cpu'
    if useCuda:
        device = 'cuda'
    if startingNeuralNetworkFilepath.upper() == 'NONE':
        startingNeuralNetworkFilepath = None

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Create the input-output pairs of filepaths
    filepaths = FilepathsUnder(validationDirectory)
    inputOutput_filepath_tuples = []
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        if heatmapKeyword in filename:
            input_img_filepath = filepath.replace(heatmapKeyword, inputImageKeyword)
            if not os.path.exists(input_img_filepath):
                raise FileNotFoundError("main(): The file {} doesn't exist".format(input_img_filepath))
            inputOutput_filepath_tuples.append((input_img_filepath, filepath))

    validation_dataset = ImageHeatmapDataset(inputOutput_filepath_tuples, image_sizeHW, preprocessing)
    train_dataset = GeneratedImageHeatmapDataset(
        foreground_images_directory=foregroundImagesDirectory,
        image_sizeHW=image_sizeHW,
        dataset_size=numberOfBatchesPerEpoch * batchSize,
        preprocessing=preprocessing,
        scale_range=scaleRange,
        rotation_range=rotationRange,
        hue_delta_range=hueDeltaRange,
        luminance_delta_range=luminanceDeltaRange,
        foreground_luminance_inverse_threshold=foregroundLuminanceInverseThreshold,
        number_of_foreground_objects=numberOfForegroundObjects)

    test_tsr, test_target_heatmap = validation_dataset[0]
    if type(test_tsr) == list:
        test_tsr = [t.unsqueeze(0).to(device) for t in test_tsr]
    else:
        test_tsr = test_tsr.unsqueeze(0).to(device)
    test_target_heatmap = test_target_heatmap.to(device)
    test_directory = os.path.join(outputDirectory, "test")
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)

    # Create data loaders
    train_dataLoader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    validation_dataLoader = DataLoader(validation_dataset, batch_size=batchSize, shuffle=True)

    # Create the neural network
    neural_network = None
    if architecture == 'Assinica_16_32_64':
        neural_network = arch.Assinica(number_of_channels=(16, 32, 64), dropout_ratio=dropoutRatio)
    elif architecture == 'Cascapedia_16_32_64':
        neural_network = arch.Cascapedia(number_of_channels=(16, 32, 64), dropout_ratio=dropoutRatio)
    elif architecture == 'Daaquam_32_64':
        neural_network = arch.Daaquam(number_of_channels=(32, 64), dropout_ratio=dropoutRatio)
    elif architecture == 'Resnet50':
        neural_network = arch.Resnet50()
    else:
        raise NotImplementedError("train_heatmap.main(): Not implemented architecture '{}'".format(architecture))

    if startingNeuralNetworkFilepath is not None:
        neural_network.load_state_dict(torch.load(startingNeuralNetworkFilepath, map_location=torch.device('cpu')))
    neural_network.to(device)

    # Optimization parameters
    criterion = nn.BCELoss()
    optimizer = optim.Adam( filter(lambda p: p.requires_grad, neural_network.parameters()), lr=learningRate, betas=(momentum, 0.999),
                           weight_decay=weightDecay)

    lowest_validation_loss = sys.float_info.max
    # Start training
    with open(os.path.join(outputDirectory, "epochLoss.csv"), 'w+') as epoch_loss_file:
        epoch_loss_file.write("Epoch,train_loss,validation_loss\n")
    for epoch in range(1, numberOfEpochs + 1):
        logging.info("*** Epoch {} ***".format(epoch))
        running_loss = 0.0
        number_of_batches = 0
        neural_network.train()
        for (i, (train_input_tsr, train_target_output_tsr)) in enumerate(train_dataLoader, 0):
            print ('.', end='', flush=True)
            if type(train_input_tsr) == list:
                train_input_tsr = [t.to(device) for t in train_input_tsr]
            else:
                train_input_tsr = train_input_tsr.to(device)
            train_target_output_tsr = train_target_output_tsr.to(device)
            optimizer.zero_grad()
            train_output_tsr = neural_network(train_input_tsr)
            loss = criterion(train_output_tsr, train_target_output_tsr)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            number_of_batches += 1
        train_average_loss = running_loss/number_of_batches

        # Validation
        with torch.no_grad():
            neural_network.eval()
            validation_running_loss = 0.0
            number_of_batches = 0
            for (valid_input_tsr, valid_target_output_tsr) in validation_dataLoader:
                if type(valid_input_tsr) == list:
                    valid_input_tsr = [t.to(device) for t in valid_input_tsr]
                else:
                    valid_input_tsr = valid_input_tsr.to(device)
                valid_target_output_tsr = valid_target_output_tsr.to(device)
                valid_output_tsr = neural_network(valid_input_tsr)
                loss = criterion(valid_output_tsr, valid_target_output_tsr)
                validation_running_loss += loss.item()
                number_of_batches += 1

            validation_average_loss = validation_running_loss / number_of_batches

            logging.info("Epoch {}; train_loss = {}; validation_loss = {}".format(epoch, train_average_loss, validation_average_loss))
            with open(os.path.join(outputDirectory, "epochLoss.csv"), 'a') as epoch_loss_file:
                epoch_loss_file.write("{},{},{}\n".format(epoch, train_average_loss, validation_average_loss))

            if validation_average_loss < lowest_validation_loss:
                lowest_validation_loss = validation_average_loss
                champion_filepath = os.path.join(outputDirectory,
                                                 "{}_ep{}_{:.4f}.pth".format(architecture, epoch, lowest_validation_loss))
                torch.save(neural_network.state_dict(), champion_filepath)

            # Test image
            test_output_tsr = neural_network(test_tsr)
            test_output_img = (test_output_tsr.squeeze().detach().cpu().numpy() * 256).astype(np.uint8)
            cv2.imwrite(os.path.join(test_directory, "heatmap_{}.png".format(epoch)), test_output_img)

def FilepathsUnder(directory):
    filepaths = [os.path.join(directory, f) for f in os.listdir(directory) \
                           if os.path.isfile(os.path.join(directory, f))]
    return filepaths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('foregroundImagesDirectory', help="The directory containing the foreground images")
    parser.add_argument('validationDirectory',
                        help="The directory containing the validation input images and their corresponding target heatmaps")
    parser.add_argument('--randomSeed', help="The random seed. Default: 0", type=int, default=0)
    parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs_train_heatmap'", default='./outputs_train_heatmap')
    parser.add_argument('--imageSizeHW', help="The size (H, W) of the generated images. Default: '(256, 256)'",
                        default='(256, 256)')
    parser.add_argument('--useCpu', help="Use CPU, even if a GPU is available", action='store_true')
    parser.add_argument('--heatmapKeyword',
                        help="A substring in the filename, indicating a target output heatmap. Default: 'heatmap_'",
                        default='heatmap_')
    parser.add_argument('--inputImageKeyword',
                        help="A substring in the filename, indicating an input image. Default: 'image_'",
                        default='image_')
    parser.add_argument('--batchSize', help="The batch size. default: 16", type=int, default=16)
    parser.add_argument('--architecture', help="The neural network architecture. Default: 'Assinica_16_32_64'", default='Assinica_16_32_64')
    parser.add_argument('--dropoutRatio', help="The dropout ratio. Default: 0.5", type=float, default=0.5)
    parser.add_argument('--learningRate', help='The learning rate. Default: 0.001', type=float, default=0.001)
    parser.add_argument('--momentum', help='The momentum. Default: 0.9', type=float, default=0.9)
    parser.add_argument('--weightDecay', help="The optimizer weight decay parameter. Default: 0.0000001", type=float,
                        default=0.0000001)
    parser.add_argument('--numberOfEpochs', help="The number of epochs. Default: 300", type=int, default=300)
    parser.add_argument('--numberOfBatchesPerEpoch', help="The number of generated batches per epoch. Default: 10", type=int, default=10)
    parser.add_argument('--startingNeuralNetworkFilepath', help="The filepath to the starting neural network. Default: 'None'", default='None')
    parser.add_argument('--preprocessing', help="The image preprocessing. Default: 'None'", default='None')
    parser.add_argument('--scaleRange',
                        help="The scale of the foreground with respect to the background size. Default: '(0.2, 0.7)'",
                        default='(0.2, 0.7)')
    parser.add_argument('--rotationRange',
                        help="The rotation range for the foreground image. in radians. Default: '(0, 6.28)'",
                        default='(0, 6.28)')
    parser.add_argument('--hueDeltaRange',
                        help="The range of hue change for the foreground image. Default: '(-15, 15)'",
                        default='(-15, 15)')
    parser.add_argument('--luminanceDeltaRange',
                        help="The range of luminance change for the foreground image. Default: '(-15, 15)'",
                        default='(-15, 15)')
    parser.add_argument('--foregroundLuminanceInverseThreshold',
                        help="The luminance inverse threshold, to segment the foreground object. Default: 220",
                        type=int, default=220)
    parser.add_argument('--numberOfForegroundObjects', help="The number of generated foreground objects. Default: 1", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.randomSeed)
    image_sizeHW = ast.literal_eval(args.imageSizeHW)
    useCuda = not args.useCpu and torch.cuda.is_available()
    preprocessing = None
    if args.preprocessing.upper() != 'NONE':
        preprocessing = args.preprocessing
    scaleRange = ast.literal_eval(args.scaleRange)
    rotationRange = ast.literal_eval(args.rotationRange)
    hueDeltaRange = ast.literal_eval(args.hueDeltaRange)
    luminanceDeltaRange = ast.literal_eval(args.luminanceDeltaRange)

    main(
        args.foregroundImagesDirectory,
        args.validationDirectory,
        args.outputDirectory,
        image_sizeHW,
        useCuda,
        args.heatmapKeyword,
        args.inputImageKeyword,
        args.batchSize,
        args.architecture,
        args.dropoutRatio,
        args.learningRate,
        args.momentum,
        args.weightDecay,
        args.numberOfEpochs,
        args.numberOfBatchesPerEpoch,
        args.startingNeuralNetworkFilepath,
        preprocessing,
        scaleRange,
        rotationRange,
        hueDeltaRange,
        luminanceDeltaRange,
        args.foregroundLuminanceInverseThreshold,
        args.numberOfForegroundObjects
    )