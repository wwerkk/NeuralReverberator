from util import convert_sample_rate, generate_specgrams

convert_sample_rate('data/MIT_IR_dataset', 'preprocessed_audio', 16000)
generate_specgrams('preprocessed_audio', 'spectrograms')