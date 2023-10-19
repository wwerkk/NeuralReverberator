from util import convert_sample_rate, generate_specgrams
import os

# Change the working directory to the train folder
if os.getcwd().split('/')[-1] != 'train':
    os.chdir('train')

convert_sample_rate('data/MIT_IR_dataset', 'preprocessed_audio', 16000)
generate_specgrams('preprocessed_audio', 'spectrograms')