import pickle
from AudioLibrary.AudioSignal import *
from AudioLibrary.AudioFeatures import *


class AudioEmotionRecognition:

    def __init__(self, model_path):

        # Load classifier
        self._clf = pickle.load(open(os.path.join(model_path, 'MODEL_CLF.p'), 'rb'))

        # Load features parameters
        self._features_param = pickle.load(open(os.path.join(model_path, 'MODEL_PARAM.p'), 'rb'))

        # Load feature scaler parametrs (mean and std)
        self._features_mean, self._features_std = pickle.load(open(os.path.join(model_path, 'MODEL_SCALER.p'), 'rb'))

        # Load PCA
        self._pca = pickle.load(open(os.path.join(model_path, 'MODEL_PCA.p'), 'rb'))

        # Load label encoder
        self._encoder = pickle.load(open(os.path.join(model_path, 'MODEL_ENCODER.p'), 'rb'))

    '''
    Function to scale audio features
    '''
    def scale_features(self, features):

        # Scaled features
        scaled_features = (features - self._features_mean) / self._features_std

        # Return scaled features
        return scaled_features

    '''
    Function to predict speech emotion from an audio signals
    '''
    def predict_emotion(self, audio_signal, predict_proba=False, decode=True):

        # Extract audio features
        audio_features = AudioFeatures(audio_signal, float(self._features_param.get("win_size")),
                                       float(self._features_param.get("win_step")))
        features, features_names = audio_features.global_feature_extraction(stats=self._features_param.get("stats"),
                                                                            features_list=self._features_param.get(
                                                                                "features_list"),
                                                                            nb_mfcc=self._features_param.get("nb_mfcc"),
                                                                            diff=self._features_param.get("diff"))
        # Scale features
        features = self.scale_features(features)

        # Apply feature dimension reduction
        if self._features_param.get("PCA") is True:
            features = self._pca.transform(features)

        # Make prediction
        if predict_proba is True:
            prediction = self._clf.predict_proba(features.reshape(1, -1))
        else:
            prediction = self._clf.predict(features.reshape(1, -1))

        # Decode label emotion
        if decode is True:
            prediction = (self._encoder.inverse_transform((prediction.astype(int).flatten())))

        # Remove gender recognition
        prediction = prediction[0][2:]

        return prediction
