import numpy as np
import pickle
import os
from scipy.signal import resample
from sklearn.model_selection import train_test_split

# Load the BVP (PPG) signal and HR labels from the WESAD .pkl file
def load_wesad_bvp_and_label_from_pkl(subject_path):
    pkl_file = os.path.join(subject_path, f'{os.path.basename(subject_path)}.pkl')
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Extract BVP signal from the wrist data
    bvp_signal = data['signal']['wrist']['BVP']

    # Extract HR labels
    hr_labels = data['label']

    return bvp_signal, hr_labels

# Downsample the PPG signal from 64Hz to 40Hz
def downsample_signal(ppg_signal, original_fs=64, target_fs=40):
    num_samples = int(len(ppg_signal) * target_fs / original_fs)
    return resample(ppg_signal, num_samples)

# Function to segment the signal with an 8s window and 2s overlap
def segment_signal(signal, window_size, overlap_size):
    num_segments = (len(signal) - window_size) // overlap_size + 1
    segments = []
    for i in range(num_segments):
        start = i * overlap_size
        end = start + window_size
        segments.append(signal[start:end])
    return np.array(segments)

# Prepare X (PPG signal) and y (HR values) for training
def prepare_data(subject_path, window_duration=8, overlap_duration=2, original_fs=64, target_fs=40):
    # Load BVP (PPG) signal and HR labels from the .pkl file
    ppg_signal, hr_labels = load_wesad_bvp_and_label_from_pkl(subject_path)

    # Downsample the PPG signal from 64Hz to 40Hz
    downsampled_ppg_signal = downsample_signal(ppg_signal, original_fs, target_fs)

    # Calculate the new window and overlap sizes in terms of samples
    window_size = window_duration * target_fs
    overlap_size = overlap_duration * target_fs

    # Segment the downsampled PPG signal
    X = segment_signal(downsampled_ppg_signal, window_size, overlap_size)

    # Align HR labels with the segments (we assume the labels are already aligned and resampled if necessary)
    y = hr_labels[:len(X)]  # Make sure the length of y matches X

    return X, y

# Process all subjects in the directory
def process_all_subjects(subjects_dir):
    all_X = []
    all_y = []

    # Loop through each subject folder (S2, S3, ..., S17)
    for subject_folder in os.listdir(subjects_dir):
        subject_path = os.path.join(subjects_dir, subject_folder)

        if os.path.isdir(subject_path):
            print(f'Processing: {subject_path}')

            # Prepare the data for the subject
            X, y = prepare_data(subject_path)

            all_X.append(X)
            all_y.append(y)

    # Combine all subjects' data
    all_X = np.vstack(all_X)
    all_y = np.hstack(all_y)

    return all_X, all_y

# Directory containing all subject folders (S2, S3, ..., S17)
subjects_dir = './'  # Replace with the actual path

# Process the data for all subjects
X, y = process_all_subjects(subjects_dir)

#X = X.squeeze(-1)
#
## Randomly divide the data into an 80:20 train-test split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#print(f'Shape of X_train: {X_train.shape}')
#print(f'Shape of X_test: {X_test.shape}')
#print(f'Shape of y_train: {y_train.shape}')
#print(f'Shape of y_test: {y_test.shape}')
#
#np.save('X_WESAD_train.npy', X_train)
#np.save('X_WESAD_test.npy', X_test)
#np.save('y_WESAD_train.npy', y_train)
#np.save('y_WESAD_test.npy', y_test)

'''
Shape of X_train: (34708, 320)
Shape of X_test: (8677, 320)
Shape of y_train: (34708,)
Shape of y_test: (8677,)
'''
