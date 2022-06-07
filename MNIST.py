import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
import tensorflow as tf

df = pd.read_csv('train.csv')
y = df['label']
df.drop('label', axis=1, inplace=True)
X = df/254

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


dummy_y = np_utils.to_categorical(y_train)

model1 = Sequential()
model1.add(Dense(784, activation='relu', input_dim=784))
model1.add(Dropout(0.2))
model1.add(Dense(523, activation='gelu'))
model1.add(Dropout(0.2))
model1.add(Dense(392, activation='gelu'))
model1.add(Dropout(0.2))
model1.add(Dense(10, activation='softmax'))
input_shape = X_train.shape
model1.build(input_shape)
model1.summary()

es = EarlyStopping(patience=6, verbose=3, monitor='loss')
model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
               metrics=tf.keras.metrics.BinaryAccuracy())
model1.fit(X_train, dummy_y,  epochs=1, batch_size=500, callbacks=es)

y_pred = model1.predict(X_test).round()
dummy_y_test = np_utils.to_categorical(y_test)
print(classification_report(dummy_y_test, y_pred))


data_to_transform = np.array(df)
X = np.reshape(data_to_transform, (42000, 28, 28, 1))
X = X / 254
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model2 = Sequential()
model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model2.add(MaxPool2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model2.add(MaxPool2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(128, activation='tanh'))
model2.add(Dropout(0.25))
model2.add(Dense(64, activation='tanh'))
model2.add(Dropout(0.25))
model2.add(Dense(10, activation='softmax'))

es = EarlyStopping(patience=6, verbose=3, monitor='loss')
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model2.summary()

model2.fit(X_train, dummy_y, epochs=1, batch_size=500, callbacks=es)
y_pred = model2.predict(X_test).round()
dummy_y_test = np_utils.to_categorical(y_test)
print(classification_report(dummy_y_test, y_pred))

for k in range(8):
    img = Image.open(fr'real_data\digit{k}.jpg')
    img = tf.image.rgb_to_grayscale(img)
    img = tf.keras.preprocessing.image.smart_resize(img, (28, 28))
    array = tf.keras.preprocessing.image.img_to_array(img)
    image = tf.keras.preprocessing.image.array_to_img(array)
    array = (array - array.min()) / (array.max() - array.min())
    array = (1 - array)
    array[array < 0.5] = 0
    data = array.reshape(1, 28, 28, 1)
    img = tf.keras.preprocessing.image.array_to_img(array)
    plt.imshow(img, cmap='gray')
    plt.show()
    y_pred = model2.predict(data).round()
    s = pd.DataFrame(y_pred)
    result = pd.DataFrame(pd.get_dummies(s).idxmax(1))
    print(f'This is {result.iloc[0].iloc[0]}')

for k in range(8):
    img = Image.open(fr'real_data\digit{k}.jpg')
    img = tf.image.rgb_to_grayscale(img)
    img = tf.keras.preprocessing.image.smart_resize(img, (28, 28))
    array = tf.keras.preprocessing.image.img_to_array(img)
    image = tf.keras.preprocessing.image.array_to_img(array)
    array = (array - array.min()) / (array.max() - array.min())
    array = (1 - array)
    array[array < 0.5] = 0
    img = tf.keras.preprocessing.image.array_to_img(array)
    plt.imshow(img, cmap='gray')
    plt.show()
    array = array.reshape((784,))
    data = pd.DataFrame(array.reshape(1, 784))
    y_pred = model1.predict(data).round()
    s = pd.DataFrame(y_pred)
    result = pd.DataFrame(pd.get_dummies(s).idxmax(1))
    print(f'This is {result.iloc[0].iloc[0]}')
