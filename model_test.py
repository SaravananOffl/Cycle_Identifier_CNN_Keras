import h5py
from keras.models import load_model
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

plt.figure(1)
model = load_model('model1.h5')

image = cv2.imread('bk.jpeg')

plt.imshow(image)
plt.title("INPUT IMAGE")

image_arr = cv2.resize(image, (300,300))
image_arr = image_arr/255.0
image_arr = np.reshape(image_arr, (-1, 300, 300, 3))

plt.figure(2)
predict_matrix = model.predict(image_arr)
print(predict_matrix)
predict =model.predict_classes(image_arr)
plt.imshow(image)
if predict ==0:
    title = "It's not a CYCLE"

else:
    title = "It's a CYCLE"
plt.title(title)
plt.show()