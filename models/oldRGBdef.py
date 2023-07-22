import torch
import numpy as np
import pandas as pd

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
        min_value = -1
        max_value = 1
        normalizedRGB = (sampleRGB - min_value) / (max_value - min_value)
        distances = torch.sqrt(torch.sum((torch.sub(cmap, normalizedRGB))**2, axis=1))
        [minDistance, minIndex] = torch.min(distances, dim=0)
        normalizedIndex = (minIndex) / (256)
        value = normalizedIndex
        return value
    
    
    #bounds for the velocity 
    boundsU = [-10.6864, 9.6914]
    #inputted data 
    img = torch.Tensor.permute(data, (1,2,0))
    
    #import the parula colormap values from .csv
    cmap = torch.tensor(pd.read_csv('parula.csv', header = None).to_numpy())

    imsize = torch.Tensor.size(img)
    ValuesRemapped = torch.zeros(torch.Tensor.size(img,0), torch.Tensor.size(img,1))
    
    # calculate the velocity for each pixel based on its RGB value
    for x in range(0,imsize[0]):
        for y in range(0,imsize[1]):
        
            sampleX = x
            sampleY = y

            normalizedValueFromRGB = lookUpRGBValue(cmap, img, sampleX, sampleY)

            physical_val = physicalFromNormalized(normalizedValueFromRGB,boundsU)
            ValuesRemapped[x,y] = normalizedValueFromRGB

    return ValuesRemapped

a = torch.load('tensor.pt')
trial = RGBtoVel(a)
b = trial