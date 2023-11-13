from keras import layers
from keras import optimizers
from keras.models import Model
## TODO: increase model capacity
## TODO: overfit

def build_spectral_regression(input_shape=(513, 256, 1), n_filters=[32, 64, 128, 256, 512], lr=0.001):
    f1, f2, f3, f4 = n_filters[:4]

    input_spect = layers.Input(input_shape)
    x = layers.Conv2D(f1, (5,5), padding='same', strides=(2,2))(input_spect)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f1, (4,4), padding='same', strides=(2,2))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f1, (4,4), padding='same', strides=(2,2))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f2, (4,4), padding='same', strides=(2,2))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f2, (4,4), padding='same', strides=(2,2))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f2, (4,4), padding='same', strides=(2,2))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f3, (4,4), padding='same', strides=(2,2))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f3, (4,4), padding='same', strides=(2,2))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f3, (4,4), padding='same', strides=(2,1))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f4, (1,1), padding='same', strides=(2,1))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)

    # Flatten and add Dense layer for regression
    x = layers.Flatten()(x)
     # Intermediate layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(7)(x)  # Output layer for 7 regression parameters

    model = Model(input_spect, output)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
    model.summary()
    
    return model