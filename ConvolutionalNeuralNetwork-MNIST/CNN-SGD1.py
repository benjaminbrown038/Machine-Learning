import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],1,28,28)
y_train = y_train.reshape(y_train.shape[0],1,28,28)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

model.add(Conv2D(8,(3,3),padding = 'same',activation = 'relu',input_shape = (1,28,28)))
model.add(MaxPool2D((2,2),padding = 'same'))
model.add(Conv2D(64,(3,3),padding='same',activation = 'relu'))
model.add(MaxPool2D((2,2),padding= 'same'))
model.add(Conv2D(128,(3,3),padding = 'same',activation = 'relu'))
model.add(MaxPool2D((2,2),padding = 'same'))
model.add(Conv2D(10,(3,3),padding = 'same',activation = 'softmax'))
model.add(MaxPool2D((4,4),padding = 'same'))
model.add(Flatten())

loss = CategoricalCrossentropy()
ac = Accuracy()
opt = SGD(lr = .01)

model.compile(loss = loss, accuracy = ac, optimizer = opt)
model.fit(x_train,y_train,
        validation_data = (x_test,y_test),
        epochs = 25,
        batch_size = 60000,
        steps_per_epoch = 1,
        validation_steps = 1,
        verbose = 2)
