import numpy as np 
from preprocess import imagelist_generator, image_labeler, imgtoarray
from keras.utils import to_categorical
from random import shuffle

def load_files(bike_folder, not_bike_folder, shuffles = True):


    bike_files = imagelist_generator(bike_folder, shuffle_files= True )
    not_bike_files = imagelist_generator(not_bike_folder, shuffle_files= True)
    files = bike_files + not_bike_files
    if(shuffles):
        shuffle(files)
    labels = image_labeler(files)
    one_hotlabels = to_categorical(labels)
    return [files, one_hotlabels]

def train_test_splitter(dataset, ratio= 0.7):
    
    total_points = len(dataset[0])
    train_size = int(ratio*total_points)
    train_x = dataset[0][:train_size]
    train_y = dataset[1][:train_size]
    test_x = dataset[0][train_size:]
    test_y = dataset[1][train_size:]
    return train_x, train_y, test_x, test_y

def image_to_memory(train_x_files, test_x_files):
    train_x =  []
    test_x = [] 
    for file in train_x_files:
        train_x.append(imgtoarray(file))
    
    for file in test_x_files:
        test_x.append(imgtoarray(file))
    
    return np.array(train_x), np.array(test_x)

def load_dataset(bike_folder, not_bike_folder):
    dataset = load_files(bike_folder, not_bike_folder)
    train_x_files, train_y, test_x_files, test_y = train_test_splitter(dataset)
    train_x, test_x = image_to_memory(train_x_files, test_x_files)
    return train_x, train_y, test_x, test_y

