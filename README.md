# Neural Decoding Model

This project implements a simple pipeline for EEG signal acquisition, preprocessing, feature extraction, and classification using a Support Vector Machine (SVM). It is intended to be a starting point for more complex EEG signal processing tasks.

## Dependencies

This project is written in Python and requires the following packages:

- numpy
- mne
- scikit-learn
- joblib

You can install these packages using pip:

```shell
pip install numpy mne scikit-learn joblib
```

## Structure

The codebase is organized into a single Python script, with the primary functions being:

- `acquire_signal()`: Loads the EEG data from a file.
- `preprocess_signal()`: Applies bandpass filtering and removes EOG artifacts from the raw EEG data.
- `extract_features()`: Computes the power spectral density from the preprocessed EEG data.
- `train_classifier()`: Trains an SVM classifier using the extracted features and provided labels.
- `main()`: The main function that ties all the processes together.

## Usage

You can run the script from the command line as follows:

```shell
python main.py
```

By default, the script uses 'sample_data.fif' as the input file, a test size of 0.2 for train/test split, and a low frequency of 1 and high frequency of 40 for the bandpass filter and power spectral density computation. You can change these parameters by modifying the `main()` function call at the end of the script:

```python
if __name__ == '__main__':
    main(file_name='your_file.fif', test_size=0.3, low_freq=1, high_freq=50)
```

The labels for training and testing the classifier are randomly generated for the purpose of this demo. In a real-life scenario, you would have to provide the labels based on your specific task.

After training, the classifier is saved to a file named 'trained_model.joblib'. You can load this model for further use as follows:

```python
from joblib import load
classifier = load('trained_model.joblib')
```

## Output

The script will print the accuracy of the classifier on the test set as follows:

```
The accuracy of the classifier is XX.XX%
```

Where XX.XX is the accuracy percentage.

