from torchvision import transforms
import torch 
import pandas as pd
from torchvision.transforms import Lambda
from torch.utils.data import Dataset

def RGBtoVel(data): #based off Christoph's MATLAB code to generate the velocities from the images for validation steps
    
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
    
    #bounds for the U velocity
    bounds = [-10.6864, 9.6914]
    #inputted data 
    img = data
    #import the parula colormap values as a tensor
    cmap = torch.tensor(pd.read_csv('parula.csv').to_numpy())

    imsize = torch.Tensor.size(img)
    ValuesRemapped = torch.zeros(torch.Tensor.size(img,0), torch.Tensor.size(img,1))

    # calculate the velocity for each pixel based on its RGB value
    for x in range(0,imsize[0]):
        for y in range(0,imsize[1]):
        
            sampleX = x
            sampleY = y

            normalizedValueFromRGB = lookUpRGBValue(cmap, img, sampleX, sampleY)

            physical_val = physicalFromNormalized(normalizedValueFromRGB,bounds)
            ValuesRemapped[x,y] = physical_val

    return ValuesRemapped

#define the transform as the custom lambda transform using the RGB to Vel def
transforms = Lambda(RGBtoVel)

#make the subset to be used in the "mass loss"
class GetVel(Dataset):
    #initialize the class, get the ground truth (real) and generated (fake) images and the transforms (custom lambda transform)
    def __init__(self, fake_data, real_data, transforms = transforms, target_transforms = transforms):
        self.real_data = torch.tensor(real_data)
        self.fake_data = torch.tensor(fake_data)
        self.transforms = transforms
        self.target_transforms = target_transforms

    #return the number of samples of the data
    def __len__ (self):
        return len(self.fake_data)
    
    def __getitem__(self, index):
        #get the ground truth and generated data for each index
        fake_data = self.fake_data[index]
        real_data = self.real_data[index]
        #apply the transforms (RGBtoVel) to the data subset
        if self.transforms:
            fake_data_U = self.transforms(fake_data)
        if self.target_transforms:
            real_data_U = self.target_transforms(real_data)
        #return the ground truth and generated velocities as tensors
        return fake_data, real_data, fake_data_U, real_data_U
    
#Use the custom dataset to transform RGB data to velocity data to be used to calculate loss
_ , _ , fake_velocities, real_velocities = GetVel()

#real_loader = dataloader(real_velocities, shuffle = false``)


