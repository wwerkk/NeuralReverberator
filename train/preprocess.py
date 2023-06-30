from util import convert_sample_rate, generate_specgrams

convert_sample_rate('train/data/balloon_verb', 'train/preprocessed_audio', 16000)
generate_specgrams('train/preprocessed_audio', 'train/spectrograms')