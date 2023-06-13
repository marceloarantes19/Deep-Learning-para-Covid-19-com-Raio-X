from keras.applications import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import Model 

class CXNet:
  @staticmethod
  def build(width, height, depth, classes):
    inputShape = (height, width, depth)
    backbone = DenseNet121(include_top=False, weights=None, input_shape=inputShape)
    backbone_out = backbone.output
    gap = GlobalAveragePooling2D(name='pooling_layer')(backbone_out)
    output = Dense(units=classes, activation='softmax', name='output_layer')(gap)
    chexnet_model = Model(inputs=backbone.input, outputs=output)
    #chexnet_model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1, momentum=0.0, decay=0.0), metrics=['accuracy'])
    chexnet_model.compile(Adam(lr=1e-3), loss='binary_crossentropy', metrics='accuracy')    
    return chexnet_model

