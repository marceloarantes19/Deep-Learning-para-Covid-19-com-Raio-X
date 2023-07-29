from keras.layers import Conv2D, Flatten, Dense, AveragePooling2D
from keras.models import Sequential
from keras.optimizers import SGD
class LeNetM:
  @staticmethod
  def build(h, w, prof, cl):
    model = Sequential()
    inputShape = (h, w, prof)
    model.add(Conv2D(20, (5,5), padding = "same", activation='relu', input_shape=inputShape))
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(50, (5,5), padding = "same", activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(units=cl, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0), metrics=['accuracy'])
    return model     