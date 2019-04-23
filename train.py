use the ImageDataGenerator class to stop overfitting..
#use the flow_from_directory method to classify the images automatically...
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('trainingSet', target_size = (28,28), batch_size = 32, class_mode = "categorical")
testing_set = test_datagen.flow_from_directory('testSet', target_size = (28,28), batch_size = 32, class_mode = "categorical")


#----------------------------------------------------------------------------------------------------
#part 1 --- Building the convolutional neural network
from keras.models import Sequential      #used in initialization
from keras.layers import Conv2D   #used for first step, convolution layer, to make images into 2D array
from keras.layers import MaxPooling2D    #used for pooling layers
from keras.layers import Flatten         #convert pool feature map into vector
from keras.layers import Dense           #used to add fully connected layer
from keras.layers import Dropout
import tensorflow as tf
#initializing the neural network
classifier = Sequential()

#convolutional layer-- changing the image using the feature map
classifier.add(Conv2D(28,kernel_size = (3,3), input_shape = (28,28,1), activation = 'relu'))

#max pooling-- reducing the size of the image matrix while keeping the features intact
classifier.add(MaxPooling2D(pool_size = (2,2)))

#convolutional layer-- changing the image using the feature map
classifier.add(Conv2D(28,kernel_size = (3,3), input_shape = (28,28,1), activation = 'relu'))

#max pooling-- reducing the size of the image matrix while keeping the features intact
classifier.add(MaxPooling2D(pool_size = (2,2)))

#flattening-- changing matrix into vector as input layer
classifier.add(Flatten())

#full-connection layer--creating an ANN(also called hidden layer)
classifier.add(Dense(128, activation = tf.nn.relu))

classifier.add(Dropout(0.2))

#output-layer
classifier.add(Dense(10,activation = "softmax"))

#compiling the CNN 
classifier.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

#-----------------------------------------------------------------------------------------------
#Part 2 -- Fitting the CNN to the images
classifier.fit_generator(training_set, samples_per_epoch =41000, nb_epoch = 25, validation_data = testing_set, nb_val_samples = 1030)

#------------------------------------------------------------------------------------------------
#part 3 --- saving the model
classifier.save("my_new_model.h5")
