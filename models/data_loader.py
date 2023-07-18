import torch 
import pandas as pd
from torchvision.transforms import Lambda
from torch.utils.data import Dataset

def RGBtoVel(data): #based off Christoph's MATLAB code to generate the velocities from the images for validation steps (/FinalSubissions_Vecchiarelli/Code/MATLAB Code/RevertImageToNumerical.m)
    
    #takes the index from lookUpRGBValue and scales it within the bounds of the velocity for the specific vector
    def physicalFromNormalized(normalized, bounds):
        custom_min = bounds[0]
        custom_max = bounds[1]
        physical_val = custom_min + (custom_max - custom_min) * normalized
        return physical_val
    
    #This function gets takes the image data as a tensor and compares it to the colormap (currently '.csv' of RGB values from MATLAB's "parula" colormap).
    #calculates the minimum distance of the pixel to a cmap value and returns that index
    def lookUpRGBValue(cmap, image, sampleX, sampleY):
        sampleRGB = torch.tensor([image[sampleX,sampleY,0],image[sampleX,sampleY,1],image[sampleX,sampleY,2]])
        min_value = 0
        max_value = 255
        normalizedRGB = (sampleRGB - min_value) / (max_value - min_value)
        distances = torch.sqrt(torch.sum((torch.sub(cmap, normalizedRGB))**2, axis=1))
        [minDistance, minIndex] = torch.min(distances, dim=0)
        normalizedIndex = (minIndex - 1) / (256 - 1)
        value = normalizedIndex
        return value
    
    #bounds for the velocity 
    boundsU = [-10.6864, 9.6914]
    #inputted data 
    img = data
    #import the parula colormap values from .csv
    cmap = torch.tensor(pd.read_csv('parula.csv').to_numpy())

    imsize = torch.Tensor.size(img)
    ValuesRemapped = torch.zeros(torch.Tensor.size(img,0), torch.Tensor.size(img,1))

    # calculate the velocity for each pixel based on its RGB value
    for x in range(0,imsize[0]):
        for y in range(0,imsize[1]):
        
            sampleX = x
            sampleY = y

            normalizedValueFromRGB = lookUpRGBValue(cmap, img, sampleX, sampleY)

            physical_val = physicalFromNormalized(normalizedValueFromRGB,boundsU)
            ValuesRemapped[x,y] = physical_val

    return ValuesRemapped

#define the transform as the custom lambda transform using the RGB to Vel def
transform = Lambda(RGBtoVel)

#make the subset to be used in the "mass loss"
class GetVelfromRGB(Dataset):
    #initialize the class, get the data from the images and the transform (custom lambda transform)
    def __init__(self, rgb_data, transform = transform):
        self.rgb_data = torch.tensor(rgb_data)
        self.transform = transform

    #return the number of samples of the data
    def __len__ (self):
        return len(self.rgb_data)
    
    def __getitem__(self, index):
        #get the data for each index
        rgb_data = self.rgb_data[index]
        #apply the transform (RGBtoVel) to the data subset
        if self.transform:
            vel_data = self.transform(rgb_data)
        #return the ground truth and generated velocities as tensors
        return vel_data