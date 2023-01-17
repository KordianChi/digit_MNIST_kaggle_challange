from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Activation
from keras.models import Model

def dense_nn_model(input_shape=784):
    
    inputs = Input(shape=input_shape)
    
    x = Dense(784)(inputs)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(520)(x)
    x = Activation('gelu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(360)(x)
    x = Activation('gelu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(10)(x)
    outputs = Activation('softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)


def conv_nn_model(input_shape=(28, 28, 1)):
    
    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(inputs)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    
    x = Conv2D(128, kernel_size=(3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    
    x = Flatten()(x)
    
    x = Dense(6272)(x)
    x = Activation('tanh')(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128)(x)
    x = Activation('tanh')(x)
    x = Dropout(0.25)(x)
    
    x = Dense(10)(x)
    outputs = Activation('softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)
    
