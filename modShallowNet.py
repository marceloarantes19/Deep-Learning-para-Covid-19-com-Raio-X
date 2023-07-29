from keras.layers import Conv2D, Activation, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD

class ShallowNet:
  @staticmethod
  def build(width, height, depth, classes):
    inputShape = (height, width, depth)
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1),metrics=['accuracy'])
    return model
