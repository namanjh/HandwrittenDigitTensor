
from keras.models import load_model
import cv2
import numpy as np
from skimage import util
from scipy import misc

model = load_model('my_new_model.h5')

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


img = cv2.imread('8img.jpg')
img = cv2.resize(img,(28,28))

'''use on the basis of input type

#img = ~img
cv2.imshow("image",img)
img = img.reshape(1,28,28,3)

classes = model.predict_classes(img)
print("The output is: ")
print (classes)
