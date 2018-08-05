# Cycle_Identifier_CNN_Keras

### A cycle image classifier based on CNN architechture. Implemented using keras with tensorflow backend.
### Image Dataset used : http://www-old.emt.tugraz.at/~pinz/data/GRAZ_02/ ( Bike and None files).

### Network Architechture:

<pre>
_________________________________________________________________
Layer (type)                 Output Shape              Param    
=================================================================
conv2d_1 (Conv2D)            (None, 300, 300, 20)      1520      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 150, 150, 20)      0         
_________________________________________________________________
activation_1 (Activation)    (None, 150, 150, 20)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 150, 150, 40)      20040     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 75, 75, 40)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 225000)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 500)               112500500 
_________________________________________________________________
activation_2 (Activation)    (None, 500)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 250)               125250    
_________________________________________________________________
activation_3 (Activation)    (None, 250)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 502       
_________________________________________________________________
activation_4 (Activation)    (None, 2)                 0         
=================================================================
Total params: 112,647,812
Trainable params: 112,647,812
Non-trainable params: 0
_________________________________________________________________
</pre>

To run this program on your computer :
1) Download the dataset from the above link.
2) Run ```python model_train.py``` and train the network .
Based on your computer memory/GPU, Tweak the architecture to fit into memory.
3) To test the network with new images use ``` python model_test.py ``` script.

