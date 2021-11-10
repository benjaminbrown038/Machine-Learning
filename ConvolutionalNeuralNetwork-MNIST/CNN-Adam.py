import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical

(x_train,y_train) , (x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],1,28,28)
x_test = x_test.reshape(x_test.shape[0],1,28,28)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

model = Sequential()
model.add(Conv2D(8,
                (3,3),
                padding = 'same',
                activation = 'relu',
                input_shape = (1,28,28)))
model.add(MaxPool2D((2,2),
                    padding='same'))
model.add(Conv2D(64,
                (3,3),
                padding = 'same',
                activation = 'relu'))
model.add(MaxPool2D((2,2),
                    padding = 'same'))
model.add(Conv2D(10,
                (3,3),
                padding = 'same',
                activation = 'softmax'))
model.add(MaxPool2D((4,4),
                    padding = 'same'))
model.add(Flatten())

loss = CategoricalCrossentropy()
ac = Accuracy()
opt = Adam()

model.compile(loss = loss,
            accuracy = ac,
            optimizer = opt)

model.fit(x_train,
            y_train,
            batch_size = 60000,
            epochs = 25,
            validation_data = (x_test,y_test)
            steps_per_epoch = 1,
            verbose = 2,
            validation_steps = 1)
