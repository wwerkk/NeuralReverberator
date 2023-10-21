from keras import layers
from keras import optimizers
from keras.models import Model

def build_spectral_ae(input_shape=(513, 256, 1),
                      latent_dim=3,
                      cond_dim=10,
                      n_filters=[32, 64, 128, 256, 512],
                      lr=0.001):

    f1, f2, f3, f4 = n_filters[:4]

    input_spect = layers.Input(input_shape)
    cond_input = layers.Input((cond_dim,))

    # Reshape and tile condition input for concatenation
    cond_input_reshaped = layers.Reshape((1, 1, cond_dim))(cond_input)
    cond_input_tiled = layers.UpSampling2D(size=(input_shape[0], input_shape[1]))(cond_input_reshaped)

    # Concatenate condition with spectrogram input
    x = layers.Concatenate(axis=-1)([input_spect, cond_input_tiled])

    # Encoder
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
    x = layers.Conv2D(latent_dim, (1,1), padding='same', strides=(1,1))(x)

    # Generate z
    z = layers.BatchNormalization()(x)

    # Decoder
    input_z = layers.Input(shape=(z.shape[1], z.shape[2], latent_dim))
    x = layers.Concatenate(axis=-1)([input_z, cond_input_reshaped])
    x = layers.Conv2DTranspose(f4, (1,1), padding='same', strides=(1,1))(input_z)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(f3, (2,2), padding='same', strides=(2,2))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(f3, (2,2), padding='same', strides=(2,2))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(f3, (2,2), padding='same', strides=(2,2))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(f2, (2,2), padding='same', strides=(2,2))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(f2, (2,2), padding='same', strides=(2,2))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(f2, (2,2), padding='same', strides=(2,2))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(f1, (2,2), padding='same', strides=(2,2))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(f1, (2,2), padding='same', strides=(2,1))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(f1, (3,1), padding='valid', strides=(2,1))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    output_spect = layers.Conv2DTranspose(1, (1,1), padding='same', strides=(1,1))(x)
    
    encoder = Model([input_spect, cond_input], z)
    encoder.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
    encoder.summary()

    decoder = Model([input_z, cond_input], output_spect)
    decoder.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
    decoder.summary()

    outputs = decoder([encoder([input_spect, cond_input]), cond_input])

    autoencoder = Model([input_spect, cond_input], outputs)
    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
    autoencoder.summary()
    
    return encoder, decoder, autoencoder