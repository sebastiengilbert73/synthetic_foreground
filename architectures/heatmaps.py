import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models.segmentation.fcn as fcn

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

class Cascapedia(nn.Module):
    def __init__(self, number_of_channels=(16, 32, 64), dropout_ratio=0.5):
        super(Cascapedia, self).__init__()
        self.batiscan = Batiscan(number_of_channels=number_of_channels, dropout_ratio=dropout_ratio)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 3, 256, 256)
        blue_output_tsr = self.batiscan(input_tsr[:, 0, :, :].unsqueeze(1))
        green_output_tsr = self.batiscan(input_tsr[:, 1, :, :].unsqueeze(1))
        red_output_tsr = self.batiscan(input_tsr[:, 2, :, :].unsqueeze(1))
        output_tsr = blue_output_tsr  # (N, 1, 256, 256)
        output_tsr = torch.minimum(output_tsr, green_output_tsr)
        output_tsr = torch.minimum(output_tsr, red_output_tsr)
        return output_tsr

class Daaquam(nn.Module):
    def __init__(self, number_of_channels=(32, 64), dropout_ratio=0.5):
        super(Daaquam, self).__init__()
        self.number_of_channels1 = number_of_channels[0]
        self.number_of_channels2 = number_of_channels[1]
        self.conv1 = nn.Conv2d(3, self.number_of_channels1, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(self.number_of_channels1, self.number_of_channels2, kernel_size=5, padding=2)
        self.batchnorm2d = nn.BatchNorm2d(self.number_of_channels1)
        self.dropout2d = nn.Dropout2d(p=dropout_ratio)
        self.convTransposed1 = nn.ConvTranspose2d(self.number_of_channels2, self.number_of_channels1, kernel_size=3,
                                                  stride=2, padding=1, output_padding=1)
        self.convTransposed2 = nn.ConvTranspose2d(self.number_of_channels1, 1, kernel_size=3,
                                                  stride=2, padding=1, output_padding=1)

    def forward(self, inputTsr):  # inputTsr.shape = (N, 3, 256, 256)
        activation1 = F.relu(F.max_pool2d(self.conv1(inputTsr), 2))  # (N, C1, 128, 128)
        activation1 = self.batchnorm2d(activation1)
        activation2 = F.relu(F.max_pool2d(self.conv2(activation1), 2))  # (N, C2, 64, 64)
        activation2 = self.dropout2d(activation2)

        activation3 = F.relu(self.convTransposed1(activation2))  # (N, C1, 128, 128)
        activation3 = self.dropout2d(activation3)
        activation4 = self.convTransposed2(activation3)  # (N, 1, 256, 256)
        output_tsr = torch.sigmoid(activation4)  # (N, 1, 256, 256)
        return output_tsr

class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.resnet50 = torchvision.models.segmentation.fcn_resnet50(pretrained=True, num_classes=21)
        # Freeze all the parameters
        for param in self.resnet50.parameters():
            param.requires_grad = False
        # Replace the heads to have 2 output channels
        self.resnet50.classifier = fcn.FCNHead(2048, 2)
        self.resnet50.aux_classifier = fcn.FCNHead(1024, 2)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 3, 256, 256)
        output_2channels_tsr = self.resnet50(input_tsr)['out']  # (N, 2, 256, 256)
        output_tsr = torch.nn.functional.softmax(output_2channels_tsr, 1)[:, 1, :, :].unsqueeze(1)  # (N, 1, 256, 256)
        return output_tsr


