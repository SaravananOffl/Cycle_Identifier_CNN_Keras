import numpy as np 
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from dataloader import load_dataset
from keras import optimizers


bike_folder  ='bike/'
not_bike_folder = 'none/'
train_x, train_y, test_x, test_y =load_dataset(bike_folder, not_bike_folder)

class Net():
    @staticmethod
    def build():
        #image dimensions
        input_dim = (300, 300, 3)
        model = Sequential()
        
        # layer1 CONV -> POOL -> ACTIVATION
        model.add(Conv2D(20, kernel_size = 5, padding = 'same',
         input_shape = input_dim))
        model.add(MaxPooling2D(pool_size= (2,2), strides = 2)) 
        model.add(Activation('relu'))

        #layer2 CONV -> POOL -> FLATTEN
        model.add(Conv2D(40, kernel_size = 5, padding = 'same'))
        model.add(MaxPooling2D(pool_size = (2,2), strides = 2))
        model.add(Flatten())

        #FULLY CONNECTED LAYERS
        model.add(Dense(500))
        model.add(Activation('relu'))

        model.add(Dense(250))
        model.add(Activation('relu'))

        # 2 class classification
        # vector [1 0] -> Not a bike
        # vector [0 1] -> It's a bike
        model.add(Dense(2))
        model.add(Activation('softmax'))
        return model 


model = Net.build()
def train(model, train_x, train_y, test_x, test_y):
    sgd = optimizers.SGD(lr = 0.01)
    print(model.summary)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy',
                   metrics = ['accuracy'] )
    model.fit(train_x, train_y, batch_size = 12, 
                validation_data = (test_x, test_y), epochs = 10)
    score = model.evaluate(test_x, test_y, batch_size = 50)
    print(f" SCORE IS {score} \n SCORE ACC IS {score[1]}  " )
    model.save('model1.h5')

train(model, train_x, train_y, test_x, test_y)
