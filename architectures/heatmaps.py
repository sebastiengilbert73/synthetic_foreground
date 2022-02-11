import torch
import torch.nn as nn
import torch.nn.functional as F

class Assinica(nn.Module):
    def __init__(self, number_of_channels=(16, 32, 64), dropout_ratio=0.5):
        super(Assinica, self).__init__()
        self.number_of_channels1 = number_of_channels[0]
        self.number_of_channels2 = number_of_channels[1]
        self.number_of_channels3 = number_of_channels[2]
        self.conv1 = nn.Conv2d(3, self.number_of_channels1, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(self.number_of_channels1, self.number_of_channels2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(self.number_of_channels2, self.number_of_channels3, kernel_size=5, padding=2)
        self.batchnorm2d = nn.BatchNorm2d(self.number_of_channels1)
        self.dropout2d = nn.Dropout2d(p=dropout_ratio)
        self.convTransposed1 = nn.ConvTranspose2d(self.number_of_channels3, self.number_of_channels2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convTransposed2 = nn.ConvTranspose2d(self.number_of_channels2, self.number_of_channels1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convTransposed3 = nn.ConvTranspose2d(self.number_of_channels1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, inputTsr):  # inputTsr.shape = (N, 3, 256, 256)
        activation1 = F.relu(F.max_pool2d(self.conv1(inputTsr), 2)) # (N, C1, 128, 128)
        activation1 = self.batchnorm2d(activation1)
        activation2 = F.relu(F.max_pool2d(self.conv2(activation1), 2))  # (N, C2, 64, 64)
        activation2 = self.dropout2d(activation2)
        activation3 = F.relu(F.max_pool2d(self.conv3(activation2), 2))  # (N, C3, 32, 32)

        activation4 = F.relu(self.convTransposed1(activation3))  # (N, C2, 64, 64)
        activation4 = self.dropout2d(activation4)
        activation5 = F.relu(self.convTransposed2(activation4))  # (N, C1, 128, 128)
        activation6 = self.convTransposed3(activation5)  # (N, 1, 256, 256)
        output_tsr = torch.sigmoid(activation6 )  # (N, 1, 256, 256)
        return output_tsr

class Batiscan(nn.Module):
    def __init__(self, number_of_channels=(16, 32, 64), dropout_ratio=0.5):
        super(Batiscan, self).__init__()
        self.number_of_channels1 = number_of_channels[0]
        self.number_of_channels2 = number_of_channels[1]
        self.number_of_channels3 = number_of_channels[2]
        self.conv1 = nn.Conv2d(1, self.number_of_channels1, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(self.number_of_channels1, self.number_of_channels2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(self.number_of_channels2, self.number_of_channels3, kernel_size=5, padding=2)
        self.batchnorm2d = nn.BatchNorm2d(self.number_of_channels1)
        self.dropout2d = nn.Dropout2d(p=dropout_ratio)
        self.convTransposed1 = nn.ConvTranspose2d(self.number_of_channels3, self.number_of_channels2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convTransposed2 = nn.ConvTranspose2d(self.number_of_channels2, self.number_of_channels1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convTransposed3 = nn.ConvTranspose2d(self.number_of_channels1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, inputTsr):  # inputTsr.shape = (N, 1, 256, 256)
        activation1 = F.relu(F.max_pool2d(self.conv1(inputTsr), 2)) # (N, C1, 128, 128)
        activation1 = self.batchnorm2d(activation1)
        activation2 = F.relu(F.max_pool2d(self.conv2(activation1), 2))  # (N, C2, 64, 64)
        activation2 = self.dropout2d(activation2)
        activation3 = F.relu(F.max_pool2d(self.conv3(activation2), 2))  # (N, C3, 32, 32)

        activation4 = F.relu(self.convTransposed1(activation3))  # (N, C2, 64, 64)
        activation4 = self.dropout2d(activation4)
        activation5 = F.relu(self.convTransposed2(activation4))  # (N, C1, 128, 128)
        activation6 = self.convTransposed3(activation5)  # (N, 1, 256, 256)
        output_tsr = torch.sigmoid(activation6 )  # (N, 1, 256, 256)
        return output_tsr