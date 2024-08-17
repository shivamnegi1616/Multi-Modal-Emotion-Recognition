# Multi-Modal-Emotion-Recognition



# 1) Speech Emotion Recognition
The aim of this section is to explore speech emotion recognition techniques from an audio recording.


## Data
The data set used for training is the **Ryerson Audio-Visual Database of Emotional Speech and Song**: https://zenodo.org/record/1188976#.XA48aC17Q1J




## Models

### SVM

Classical approach for Speech Emotion Recognition consists in applying a series of filters on the audio signal and partitioning it into several windows (fixed size and time-step). Then, features from time domain (**Zero Crossing Rate, Energy** and **Entropy of Energy**) and frequency domain (**Spectral entropy, centroid, spread, flux, rolloff** and **MFCCs**) are extracted for each frame. We compute then the first derivatives of each of those features to capture frame to frame changes in the signal. Finally, we calculate the following global statistics on these features: *mean, median, standard deviation, kurtosis, skewness, 1% percentile, 99% percentile, min, max* and *range* and train a simple SVM classifier with rbf kernel to predict the emotion detected in the voice.


SVM classification pipeline:
- Voice recording
- Audio signal discretization
- Apply pre-emphasis filter
- Framing using a rolling window
- Apply Hamming filter
- Feature extraction
- Compute global statistics
- Make a prediction using our pre-trained model


### TimeDistributed CNNs

The main idea of a **Time Distributed Convolutional Neural Network** is to apply a rolling window (fixed size and time-step) all along the log-mel-spectrogram. Each of these windows will be the entry of a convolutional neural network, composed by four Local Feature Learning Blocks (LFLBs) and the output of each of these convolutional networks will be fed into a recurrent neural network composed by 2 cells LSTM (Long Short Term Memory) to learn the long-term contextual dependencies. Finally, a fully connected layer with *softmax* activation is used to predict the emotion detected in the voice.


TimeDistributed CNNs pipeline:
- Voice recording
- Audio signal discretization
- Log-mel-spectrogram extraction
- Split spectrogram with a rolling window
- Make a prediction using our pre-trained model
