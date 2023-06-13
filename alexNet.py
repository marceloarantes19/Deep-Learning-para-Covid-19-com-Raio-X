from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
class AlexNet:
  @staticmethod
  def build(width, height, depth, classes):
    inputShape = (height, width, depth)
    model = Sequential()
    model.add(Conv2D(96, (11,11), strides=(4,4), padding='same', activation='relu', kernel_initializer='uniform', input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same'))
    model.add(Conv2D(256,(5,5),strides=(1,1),padding='same', activation="relu",kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same'))
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same', activation="relu",kernel_initializer='uniform'))
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same', activation="relu",kernel_initializer='uniform'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same', activation="relu",kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same'))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(classes, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1, momentum=0.0, decay=0.0), metrics=['accuracy'])
    return model