from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Activation, BatchNormalization
from keras.models import Sequential
from keras.applications.efficientnet import EfficientNetB3
from keras import regularizers
from keras.optimizers import SGD
class EfficientNetB3M:
  @staticmethod
  def build(width, height, depth, classes):
    inputShape = (height, width, depth)
    model = Sequential()
    model.add(Conv2D(96, (11,11), strides=(4,4), padding='same', activation='swish', kernel_initializer='uniform', input_shape=inputShape))
    model.add(Dense(units = 200, activation='relu'))
    model.add(Dense(units = 200, activation = 'relu'))
    model.add(Dense(units = 2, activation='sigmoid'))
#    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1, momentum=0.0, decay=0.0), metrics=['accuracy'])

    return model    
