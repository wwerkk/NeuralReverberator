import datetime
import util
import model

# hyperparameters
epochs = 200
batch_size = 8
learning_rate = 0.0001
latent_dim = 3
n_filters = [32, 64, 128, 1024]
n_samples = None

# input
input_shape = (513, 128, 1)
rate = 16000

# build the model
e, d, ae = model.build_spectral_ae(input_shape=input_shape, 
                                   latent_dim=latent_dim,
                                   n_filters=n_filters,
                                   lr=learning_rate)

# load the data
x_train, x_test = util.load_specgrams('spectrograms', (513, 128), train_split=0.8)

start_time = datetime.datetime.today()

# train the thing
history = ae.fit(x=x_train, y=x_train,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, x_test))

end_time = datetime.datetime.today()

print(f"Completed {epochs} in {end_time-start_time}.")

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
