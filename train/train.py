import datetime
import util
import model
import os

# hyperparameters
epochs = 200
batch_size = 8
learning_rate = 0.0001
n_filters = [32, 64, 128, 1024]
n_samples = None

# input
input_shape = (513, 128, 1)
rate = 16000

# build the model
regression_model = model.build_spectral_regression(input_shape=input_shape, 
                                   n_filters=n_filters,
                                   lr=learning_rate)


# Change the working directory to the train folder
if os.getcwd().split('/')[-1] != 'train':
    os.chdir('train')

# load the data
print(os.getcwd())
x_train, x_test = util.load_specgrams('data/spectrograms', (513, 128), train_split=0.8)
y_train, y_test = util.load_params('data/params', train_split=0.8)

start_time = datetime.datetime.today()

# train the thing
history = regression_model.fit(x=x_train, y=y_train,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, y_test))

end_time = datetime.datetime.today()

print(f"Completed {epochs} in {end_time-start_time}.")

# generate report here
r = {'start_time' : start_time,
     'end_time' : end_time,
     'history' : history.history,
     'batch_size' : batch_size,
     'epochs' : epochs,
     'learning_rate' : learning_rate,
     'n_filters' : n_filters,
     'input_shape' : input_shape,
     'rate' : rate,
     'n_samples' : n_samples,
     'model' : regression_model}

util.generate_report(r)
