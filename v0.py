import numpy as np
import mne
from sklearn import svm
from sklearn.model_selection import train_test_split
from joblib import dump, load
import os

def acquire_signal(file_name):
    """
    This function reads the EEG data from a file.
    """
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"{file_name} not found.")
        
    raw_data = mne.io.read_raw_fif(file_name, preload=True)
    return raw_data

def preprocess_signal(raw_data):
    """
    This function applies bandpass filtering and removes EOG artifacts.
    """
    raw_data.filter(l_freq=1, h_freq=40)

    eog_events = mne.preprocessing.find_eog_events(raw_data)
    n_bads, scores = mne.preprocessing.ica_find_bads_eog(raw_data, eog_events)
    preprocessed_data = mne.preprocessing.ICA(n_components=0.95).fit(raw_data).apply(raw_data, exclude=n_bads)

    return preprocessed_data

def extract_features(preprocessed_data):
    """
    This function computes the power spectral density as features.
    """
    power_spectral_density, frequencies = mne.time_frequency.psd_multitaper(preprocessed_data, fmin=1, fmax=40)
    return power_spectral_density

def train_classifier(features, labels):
    """
    This function trains a Support Vector Machine classifier.
    """
    classifier = svm.SVC()
    classifier.fit(features, labels)
    return classifier

def main(file_name='sample_data.fif', test_size=0.2, low_freq=1, high_freq=40):
    """
    This function is the main function that ties all the processes together.
    """
    raw_data = acquire_signal(file_name)
    preprocessed_data = preprocess_signal(raw_data)

    features = extract_features(preprocessed_data)

    labels = np.random.randint(2, size=len(features))

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)

    classifier = train_classifier(X_train, y_train)

    # Save the model
    dump(classifier, 'trained_model.joblib')

    score = classifier.score(X_test, y_test)

    print(f'The accuracy of the classifier is {score*100:.2f}%')

if __name__ == '__main__':
    main()
