from keras.datasets import mnist
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1) / 255.0
X_test = X_test.reshape(10000, 28, 28, 1) / 255.0

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

conv_layer1 = Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), input_shape=(28, 28, 1), padding='valid',
                     activation='relu')
pooling_layer1 = AveragePooling2D(pool_size=(2, 2))
conv_layer2 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')
pooling_layer2 = AveragePooling2D(pool_size=(2, 2))

model = Sequential()
model.add(conv_layer1)
model.add(pooling_layer1)
model.add(conv_layer2)
model.add(pooling_layer2)
model.add(Flatten())

hidden_layer1 = Dense(units=120, activation='relu')
hidden_layer2 = Dense(units=84, activation='relu')
output_layer = Dense(units=10, activation='softmax')

model.add(hidden_layer1)
model.add(hidden_layer2)
model.add(output_layer)

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05), metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=5000, batch_size=4096)
model.save('my_model')

loss, accuracy = model.evaluate(X_test, Y_test)
print('Loss: ', loss)
print('Accuracy: ', accuracy)
