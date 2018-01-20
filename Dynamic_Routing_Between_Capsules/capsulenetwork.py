from keras.utils import to_categorical
import numpy as np
from keras.datasets import mnist
from keras import layers, models
from params import *
from customlayers import *


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


def build_graph():
    x = layers.Input(input_shape)
    y = layers.Input([10])
    conv1 = layers.Conv2D(**conv1_params)(x)
    conv2 = layers.Conv2D(**conv2_params)(conv1)
    conv2_reshaped = layers.Reshape(target_shape=primary_capsules_shape)(conv2)
    primary_capsules = layers.Lambda(squash, name='primary_capsules')(conv2_reshaped)
    digits_capsules = Capsule(**digits_capsules_params)(primary_capsules)
    lengths = Length(name='capsnet')(digits_capsules)
    musked_capsules = Mask(name='mask')([digits_capsules, y])
    decoder = layers.Dense(dense1, activation='relu')(musked_capsules)
    decoder = layers.Dense(dense2, activation='relu')(decoder)
    decoder = layers.Dense(np.prod(input_shape), activation='sigmoid')(decoder)
    decoder_output = layers.Reshape(target_shape=input_shape, name='decoder_output')(decoder)
    model = models.Model(inputs=[x, y], outputs=[lengths, decoder_output])
    return model


def margin_loss(y_true, y_pred):
    loss = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(loss, 1))


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    model = build_graph()
    model.compile(optimizer='adam',
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1, reconstruction_loss_coefficient],
                  metrics={'capsnet': 'accuracy'})
    model.fit([x_train, y_train], [y_train, x_train], batch_size=batch_size, epochs=3)
