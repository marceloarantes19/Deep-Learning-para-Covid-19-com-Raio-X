from keras.layers import Conv2D, Flatten, Dense, AveragePooling2D
from keras.models import Sequential
from keras.optimizers import SGD
class LeNet5M:
  @staticmethod
  def build(h, w, prof, cl):
    model = Sequential()
    inputShape = (h, w, prof)
    model.add(Conv2D(filters=6, strides=(1,1), kernel_size=(5,5), activation='tanh', input_shape=inputShape))
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(filters=6, strides=(1,1), kernel_size=(5,5), activation='tanh'))
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dense(units=120, activation='tanh'))
    model.add(Flatten())
    model.add(Dense(units=84, activation='tanh'))
    model.add(Dense(units=cl, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1, momentum=0.0, decay=0.0), metrics=['accuracy'])
    return model     