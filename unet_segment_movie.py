# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:39:23 2023
@author: liamj
"""

from unet_segment import unet_segment
import os
import cv2
import glob

# images_path = r'C:\Users\liamj\Desktop\Segmenting Neural Network\Testing Sets\Nuclei\2023_05_04_V2'
# images_path = r'E:\Nucleus Project\control\movie1\NN_images'

z_range = 20 # update to match number of slices in each movie

def main(images_path, checkpoint_path):
    # get current directory:
    od = os.getcwd()
    # navigate to image location:
    os.chdir(images_path)
    
    # get list of images in directory:
    image_list = glob.glob('./*.tif')
    directory = "SegmentationData"
    parent_dir = images_path
    path = os.path.join(parent_dir, directory)
    
    # if SegmentationData folder doesn't exist, create it:
    if os.path.exists(path) is False:
        os.mkdir(path)
    
    # loop through time and z, segmenting each image:
    os.chdir(path)
    for z in range(1,z_range):
        for i in range(1, round(len(image_list)/z_range)):
            
            # directory = f'frame{i}_z{z}'
            directory = f'frame{i:0>{4}}_z{z:0>{2}}'
            parent_dir = path
            path2 = os.path.join(parent_dir, directory)
            if os.path.exists(path2) is False:
                os.mkdir(path2)
            os.chdir(path2)
            
            im_path = os.path.join(images_path, image_list[i-1])
            seg_output = unet_segment(im_path)
            cv2.imwrite('ImageSegment.tif', seg_output)
            
            os.chdir(parent_dir)
    
    os.chdir(od)
