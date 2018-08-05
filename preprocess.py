import cv2 
import numpy as np 
import os 
from imutils import paths
import random 


def imagelist_generator(directory, shuffle_files = False, seed_no = 3, batches = False, batch_size = 0 )  :
    
    if( batches and batch_size == 0):
        print("Aborted! Batch size is not entered")
        return None
    random.seed(seed_no)
    files = list(paths.list_images(directory))
    
    if(shuffle_files):
        random.shuffle(files)

    if(batches):
        return files[:batch_size]
    
    return files

def image_labeler(files, dictionary_format = False):
    # NAMING CONVENTION
    # bike = 1 , not_bike = 0
    labels = [1 if "bike" in file else 0 for file in files]
    if(dictionary_format):
        data = list(zip(files, labels))
        return data
    return labels 

def imgtoarray(image):
    
    image_arr = cv2.imread(image)
    image_arr = cv2.resize(image_arr, (300,300))
    image_arr = image_arr/255.0
    return image_arr
