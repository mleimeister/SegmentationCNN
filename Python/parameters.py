# thresholding value for prediction-choice algorithm.  trade recall for accuracy here.
prediction_threshold = 0.3

# should we include (MLS, SSLM, beat #) features when training?
#training_features = {'mls', 'sslm', 'beat_numbers'}
training_features = {'mls', 'beat_numbers'}

# how many beats make up a context window for the MLS part of the network
context_length = 115

# number of Mel bands
num_mel_bands = 80

# how many frames to max-pool in building the SSLM
max_pool = 2

# how far back to calculate the SSLM (note that actual length will be max_pool * sslm_length)
sslm_length = 115

# how many more negative examples than segment boundaries
neg_frames_factor = 5

# oversample positive frames because there are too few
pos_frames_oversample = 5

# oversample frames between segments
mid_frames_oversample = 3

# how many frames are semi-positive examples around an annotation
label_smearing = 1

padding_length = int(context_length / 2)

