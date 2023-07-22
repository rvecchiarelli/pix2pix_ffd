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
    def lookUpRGBValue(cmap, image):
        sampleRGB = image.permute(2,0,1)
        min_value = -1
        max_value = 1
        normalizedRGB = (sampleRGB - min_value) / (max_value - min_value)
        normalizedRGB = normalizedRGB.permute(1, 2, 0)
        normalizedRGB = normalizedRGB.unsqueeze(0)
        normalizedRGB = normalizedRGB.repeat(256, 1, 1, 1)
        sub = cmap - normalizedRGB
        distances = torch.sqrt(torch.sum((cmap - normalizedRGB)**2, axis =3))
        [minVals,minIndex] = torch.min(distances, dim=(1))
        normalizedIndex = minIndex/ (255)
        return normalizedIndex
    
    
    #bounds for the velocity 
    boundsU = [-10.6864, 9.6914]
    #inputted data 
    img = torch.Tensor.permute(data, (1,2,0))
    
    #import the parula colormap values from .csv
    cmap = torch.tensor(pd.read_csv('parula.csv', header= None).values)


    normalizedValueFromRGB = lookUpRGBValue(cmap, img)
    ValuesRemapped = physicalFromNormalized(normalizedValueFromRGB,boundsU)

    return normalizedValueFromRGB
    
a = torch.load('tensor.pt')
trial = RGBtoVel(a)
b = trial
