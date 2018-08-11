## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import numpy as np


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # 1 x 224 x 224
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        I.xavier_uniform_(self.conv1.weight, gain=np.sqrt(2))
        I.constant_(self.conv1.bias, 0)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        I.xavier_uniform_(self.conv2.weight, gain=np.sqrt(2))
        I.constant_(self.conv2.bias, 0)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        self.drop1 = nn.Dropout2d(p=0.2)
        
        # pool w = 2, s = 2
        # 32 x 112 x 112
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        I.xavier_uniform_(self.conv3.weight, gain=np.sqrt(2))
        I.constant_(self.conv3.bias, 0)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        I.xavier_uniform_(self.conv4.weight, gain=np.sqrt(2))
        I.constant_(self.conv4.bias, 0)
           
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.zero_()
        self.drop2 = nn.Dropout2d(p=0.2)
        
        # pool w = 2, s = 2
        # 64 x 56 x 56
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        I.xavier_uniform_(self.conv5.weight, gain=np.sqrt(2))
        I.constant_(self.conv5.bias, 0)
        
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        I.xavier_uniform_(self.conv6.weight, gain=np.sqrt(2))
        I.constant_(self.conv6.bias, 0)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn3.weight.data.fill_(1)
        self.bn3.bias.data.zero_()
        self.drop3 = nn.Dropout2d(p=0.2)
        
        # pool w = 2, s = 2
        # 128 x 28 x 28
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        I.xavier_uniform_(self.conv7.weight, gain=np.sqrt(2))
        I.constant_(self.conv7.bias, 0)
        
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        I.xavier_uniform_(self.conv8.weight, gain=np.sqrt(2))
        I.constant_(self.conv8.bias, 0)
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn4.weight.data.fill_(1)
        self.bn4.bias.data.zero_()
        self.drop4 = nn.Dropout2d(p=0.2)
        
        # pool w = 2, s = 2
        # 256 x 14 x 14
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        I.xavier_uniform_(self.conv9.weight, gain=np.sqrt(2))
        I.constant_(self.conv9.bias, 0)
        
        self.conv0 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        I.xavier_uniform_(self.conv0.weight, gain=np.sqrt(2))
        I.constant_(self.conv0.bias, 0)
        
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn5.weight.data.fill_(1)
        self.bn5.bias.data.zero_()
        self.drop5 = nn.Dropout2d(p=0.2)
        
        # 256 x 7 x 7
        self.fc1 = nn.Linear(256*7*7, 1024)
        I.xavier_uniform_(self.fc1.weight, gain=np.sqrt(2.0))
        self.fc1_drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        I.xavier_uniform_(self.fc2.weight, gain=np.sqrt(2.0))
        self.fc2_drop = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(1024, 1024)
        I.xavier_uniform_(self.fc3.weight, gain=np.sqrt(2.0))
        self.fc3_drop = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(1024, 136)
        I.xavier_uniform_(self.fc4.weight, gain=np.sqrt(2.0))
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(self.bn1(self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))))
        x = self.drop2(self.bn2(self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))))
        x = self.drop3(self.bn3(self.pool3(F.relu(self.conv6(F.relu(self.conv5(x)))))))
        x = self.drop4(self.bn4(self.pool4(F.relu(self.conv8(F.relu(self.conv7(x)))))))
        x = self.drop5(self.bn5(self.pool5(F.relu(self.conv0(F.relu(self.conv9(x)))))))
        
        x = x.view(-1, 256*7*7)
        x = self.fc1_drop(F.relu(self.fc1(x)))
        x = self.fc2_drop(F.relu(self.fc2(x)))
        x = self.fc3_drop(F.relu(self.fc3(x)))
        x = self.fc4(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
