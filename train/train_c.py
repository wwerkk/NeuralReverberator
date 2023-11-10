import datetime
import util
import model_c
import os
from keras.callbacks import EarlyStopping, Callback

# custom callback for checkpoint save
class CustomModelCheckpoint(Callback):
    def __init__(self, encoder, decoder, checkpoint_dir):
        self.encoder = encoder
        self.decoder = decoder
        self.checkpoint_dir = checkpoint_dir
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.encoder.save(os.path.join(self.checkpoint_dir, "encoder_best.h5"))
            self.decoder.save(os.path.join(self.checkpoint_dir, "decoder_best.h5"))
            self.model.save(os.path.join(self.checkpoint_dir, "ae_best.h5"))
            print(f"\nSaved to \"{self.checkpoint_dir}\" at epoch {epoch}")

# hyperparameters
epochs = 300
batch_size = 8
learning_rate = 0.0001
latent_dim = 3
cond_dim = 32
n_filters = [32, 64, 128, 1024]
train_split = 0.8
n_samples = None

# input
input_shape = (513, 128, 1)
rate = 16000

# build the model
e, d, ae = model_c.build_spectral_ae(input_shape=input_shape, 
                                   cond_dim=cond_dim,
                                   latent_dim=latent_dim,
                                   n_filters=n_filters,
                                   lr=learning_rate)

# Change the working directory to the train folder
if os.getcwd().split('/')[-1] != 'train':
    os.chdir('train')

# load the data
x_train, x_test = util.load_specgrams('spectrograms', (513, 128), train_split=train_split, n_samples=n_samples)
p_train, p_test = util.load_params('params', train_split=train_split, n_samples=n_samples)

# define callbacks
callbacks = [
     EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10),
     CustomModelCheckpoint(encoder=e, decoder=d, checkpoint_dir='checkpoints')

]

start_time = datetime.datetime.today()

# train the thing
history = ae.fit(x=[x_train, p_train], y=[x_train, p_train],
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([x_train, p_train], [x_train, p_train]),
                callbacks=callbacks)

end_time = datetime.datetime.today()

print(f"Completed in {end_time-start_time}.")

# generate report here
r = {'start_time' : start_time,
     'end_time' : end_time,
     'history' : history.history,
     'batch_size' : batch_size,
     'epochs' : epochs,
     'learning_rate' : learning_rate,
     'latent_dim' : latent_dim,
     'n_filters' : n_filters,
     'input_shape' : input_shape,
     'rate' : rate,
     'n_samples' : n_samples,
     'encoder' : e,
     'decoder' : d,
     'autoencoder' : ae}

util.generate_report(r)