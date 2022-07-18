from PIL import Image
import numpy as np 
import cv2
import tensorflow 
import tensorflow.keras.preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten,Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image = 'mnist_png/training/0/1.png'
training = 'mnist_png/training'
testing = 'mnist_png/testing'

# load 
def load_data(file_path):
    file_path = Path(file_path)
    stacked_data = np.concatenate((np.array([(Image.open(i)) for i in file_path])))
    return stacked_data

# load 
def DataFromFile(file_path,**kwargs):
    img = tensorflow.keras.preprocessing.image.ImageDataGenerator()
    data = img.flow_from_directory(file_path,**kwargs)
    return data

# load & augment
def Image_Data_Generator2(file_path,**kwargs):
    img = tensorflow.keras.preprocessing.image.ImageDataGenerator(**kwargs)
    data = img.flow_from_directory(file_path,**kwargs)
    return data
    x_train = (np.array([data[0][0]]))
    return x_train
    y_train = (np.array(data[0][1]))
    return y_train

# transform 
def image_to_array(image):
    array = tensorflow.keras.preprocessing.image.img_to_array(image_to_array)
    return array

# transform 
def array_to_image(array):
    image = tensorflow.keras.preprocessing.image.array_to_img(image)
    return image

# tranforming
def x_train_handling(image_generator_object):
    length = len(image_generator_object)
    mini_batch = (image_generator_object[0][0].shape)[0]
    list1 = []
    for i in range(length):
        for j in range(mini_batch):
            new_array = image_generator_object[i][0][j]
            array = np.expand_dims(new_array,axis=0)
            list1.append(array)
    data = np.concatenate(list1)
    return data

# transforming
def y_train_handling(image_generator_object):
    length = len(image_generator_object)
    mini_batch = (image_generator_object[0][0].shape)[0]
    list1 = []
    for i in range(length):
        for j in range(mini_batch):
            new_array = image_generator_object[i][0][j]
            array = np.expand_dims(new_array,axis=0)
            list1.append(array)
    data = np.concatenate(list1)
    return data

# transforming
def augm_parms(**kwargs):
    datagen = ImageDataGenerator(**kwargs)
    return datagen
    
# transforming
def augm_batch(data,**kwargs):
    new_batch = datagen.flow(data)
    return data 
    
    
# handling
def show_images(image):
    cv2.imshow('Image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows('Image')

    
# handling
def load_data(file_path):
    file_path = Path(file_path)
    stacked_data = np.concatenate((np.array([(Image.open(i)) for i in file_path])))
    return stacked_data

# handling 
def Image_Data_Generator(file_path,**kwargs):
    img = tensorflow.keras.preprocessing.image.ImageDataGenerator()
    data = img.flow_from_directory(file_path,**kwargs)
    return data


# handling
def image_to_array(image):
    array = tensorflow.keras.preprocessing.image.img_to_array(image)
    return array

# handling
def array_to_image(array):
    image = tensorflow.keras.preprocessing.image.array_to_img(array)
    return image

# handling & transforming
def x_train_augment(image_generator_object):
    length = len(image_generator_object)
    mini_batch = (image_generator_object[0][0].shape)[0]
    list1 = []
    for i in range(length):
        for j in range(mini_batch):
            new_array = image_generator_object[i][0][j]
            array = np.expand_dims(new_array,axis=0)
            list1.append(array)
    data = np.concatenate(list1)
    return data


# handling and transforming 
def y_train_augment(image_generator_object):
    length = len(image_generator_object)
    mini_batch = (image_generator_object[0][0].shape)[0]
    list1 = []
    for i in range(length):
        for j in range(mini_batch):
            new_array = image_generator_object[i][0][j]
            array = np.expand_dims(new_array,axis=0)
            list1.append(array)
    data = np.concatenate(list1)
    return data
    
    

# running functions
test_gen = ImageDataGenerator()
test = test_gen.flow_from_directory(testing,color_mode='grayscale', target_size=(28,28),class_mode='categorical',  batch_size=10000)
test_example=test[0][0]
test_example /= 255
test_example = np.rollaxis(test_example,3,1)
test_dependent=test[0][1]



# running functions
train_datagen = ImageDataGenerator(vertical_flip=True)
train_generator = train_datagen.flow_from_directory(training,color_mode='grayscale',target_size=(28, 28),batch_size=60000,class_mode='categorical')
train_x=train_generator[0][0]
train_x /=255
train_x = np.rollaxis(train_x,3,1)
train_y=train_generator[0][1]
