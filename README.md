## Segmentation CNN

### Method
Convolutional neural networks (CNN) for music segmentation. Similar than in [1], a log-scaled Mel spectrogram is extracted from the audio signal, with the difference that input spectrograms are max pooled across beat times. Beat tracking was done using the [MADMOM toolbox] (https://github.com/CPJKU/madmom) with the DBN beat tracking algorithm from [2]. Context windows of 16 bars are then classified by a CNN to determine whether the central beat is a segment boundary. The CNN training was implemented using [Keras](http://keras.io/).

On the 'Internet Archive' portion of the [SALAMI](https://ddmal.music.mcgill.ca/research/salami/annotations) dataset it achieves a boundary detection f-Measure of 59% at a tolerance of 2 beats for a random 0.9/0.1 split. Some audio files did not have a corresponding annotation and were discarded.

An example of a beat-wise log Mel spectrogram
![alt text](https://github.com/mleimeister/SegmentationCNN/blob/master/Results/1279_spec.png "")
and corresponding prediction with ground truth segment annotations.
![alt text](https://github.com/mleimeister/SegmentationCNN/blob/master/Results/1279.png "")

Some more example outputs of the CNN with corresponding ground truth annotations can be found in the 'Results' subfolder (the nicer ones :)

### TODO 
This is work in progress! So far the feature extraction and evaluation was run in MATLAB, whereas for the CNN training, the Keras Python library was used. Evaluation is done on the beat level using the beat-level labels constructed from the ground truth annoations. For computing the f-Measure, the [Beat Tracking Evaluation Toolbox](https://code.soundsoftware.ac.uk/projects/beat-evaluation/) was used. Currently porting the feature extraction and evaluation to Python.

### Requirements

For the CNN training:

* [Keras](http://keras.io/)
* [Tensorflow](https://www.tensorflow.org/) or [Theano](http://deeplearning.net/software/theano/)

Feature extraction in Python:
* [librosa](https://github.com/librosa/librosa)
* Numpy, Scipy

Evaluation:
* [mir_eval](https://github.com/craffel/mir_eval)
* [peakutils](https://pypi.python.org/pypi/PeakUtils)
* Matplotlib


### References

[1] Karen Ullrich, Jan Schlüter and Thomas Grill: Boundary detection in music structure analysis using convolutional neural networks. ISMIR 2014. [pdf](http://www.ofai.at/~jan.schlueter/pubs/2014_ismir.pdf)

[2] Sebastian Böck, Florian Krebs and Gerhard Widmer, A Multi-Model Approach to Beat Tracking Considering Heterogeneous Music Styles. ISMIR 2014. [pdf](http://www.terasoft.com.tw/conf/ismir2014/proceedings/T108_367_Paper.pdf)



