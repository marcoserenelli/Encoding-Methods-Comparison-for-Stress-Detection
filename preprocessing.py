import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle
import os
from imblearn.combine import SMOTETomek
from scipy.signal import butter, filtfilt

SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
CHEST_FREQ = 700
WRIST_FREQ = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4}
MULTICLASS_LABELS = {'Baseline': 1, 'Stress': 2, 'Amusement': 3, 'Meditation': 4}


def resample_frequency(data, freq_in, freq_out):
    n_in = len(data)
    n_out = int(n_in * freq_out / freq_in)
    data_resampled = np.interp(np.linspace(0, n_in, n_out), np.arange(n_in), data)
    return data_resampled


def resample_labels(labels, freq_in, freq_out):
    n_in = len(labels)
    n_out = int(n_in * freq_out / freq_in)
    indices = np.round(np.linspace(0, n_in - 1, n_out)).astype(int)
    resampled_labels = labels[indices]
    return resampled_labels


def convert_labels(data, target_labels, binary=False):
    print('Converting labels to match target labels')
    data = data[data['label'].isin(target_labels)]
    return data


def data_normalizer(df):
    print('Normalizing data')
    features_to_ignore = ['timestamp', 'label']
    df_saved = df[features_to_ignore]
    scaler = MinMaxScaler(feature_range=(-0.99, 0.99))
    df = df.drop(columns=features_to_ignore)
    df_normalized = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    df_normalized = pd.concat([df_normalized, df_saved], axis=1)
    return df_normalized


def hampel_filter(data, window_size):
    print('Applying Hampel filter')
    columns = data.columns
    rolling = data[columns].rolling(window=window_size, center=True)
    z_score = abs(data[columns] - rolling.mean()) / rolling.std()
    z_score = z_score.fillna(-float('inf'))
    data[columns] = data[columns].mask(z_score > 3)
    data = data.interpolate(method='nearest').bfill().ffill()
    return data


def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def process_wrist_data(data, frequency, cutoff=0.3):
    print('Processing wrist data')
    wrist_acc_x = data['signal']['wrist']['ACC'][:, 0]
    wrist_acc_y = data['signal']['wrist']['ACC'][:, 1]
    wrist_acc_z = data['signal']['wrist']['ACC'][:, 2]
    wrist_bvp = data['signal']['wrist']['BVP'].flatten()
    wrist_eda = data['signal']['wrist']['EDA'].flatten()
    wrist_temp = data['signal']['wrist']['TEMP'].flatten()

    wrist_acc_x = resample_frequency(wrist_acc_x, WRIST_FREQ['ACC'], frequency)
    wrist_acc_y = resample_frequency(wrist_acc_y, WRIST_FREQ['ACC'], frequency)
    wrist_acc_z = resample_frequency(wrist_acc_z, WRIST_FREQ['ACC'], frequency)
    wrist_bvp = resample_frequency(wrist_bvp, WRIST_FREQ['BVP'], frequency)
    wrist_eda = resample_frequency(wrist_eda, WRIST_FREQ['EDA'], frequency)
    wrist_temp = resample_frequency(wrist_temp, WRIST_FREQ['TEMP'], frequency)

    # Apply Butterworth filter
    wrist_acc_x = butter_lowpass_filter(wrist_acc_x, cutoff, frequency)
    wrist_acc_y = butter_lowpass_filter(wrist_acc_y, cutoff, frequency)
    wrist_acc_z = butter_lowpass_filter(wrist_acc_z, cutoff, frequency)
    wrist_bvp = butter_lowpass_filter(wrist_bvp, cutoff, frequency)
    wrist_eda = butter_lowpass_filter(wrist_eda, cutoff, frequency)
    wrist_temp = butter_lowpass_filter(wrist_temp, cutoff, frequency)

    df = pd.DataFrame({
        'wrist_acc_x': wrist_acc_x,
        'wrist_acc_y': wrist_acc_y,
        'wrist_acc_z': wrist_acc_z,
        'wrist_bvp': wrist_bvp,
        'wrist_eda': wrist_eda,
        'wrist_temp': wrist_temp,
    })

    return df


def process_chest_data(data, frequency, cutoff=0.3):
    print('Processing chest data')
    chest_acc_x = data['signal']['chest']['ACC'][:, 0]
    chest_acc_y = data['signal']['chest']['ACC'][:, 1]
    chest_acc_z = data['signal']['chest']['ACC'][:, 2]
    chest_ecg = data['signal']['chest']['ECG'].flatten()
    chest_emg = data['signal']['chest']['EMG'].flatten()
    chest_eda = data['signal']['chest']['EDA'].flatten()
    chest_temp = data['signal']['chest']['Temp'].flatten()
    chest_resp = data['signal']['chest']['Resp'].flatten()

    chest_acc_x = resample_frequency(chest_acc_x, CHEST_FREQ, frequency)
    chest_acc_y = resample_frequency(chest_acc_y, CHEST_FREQ, frequency)
    chest_acc_z = resample_frequency(chest_acc_z, CHEST_FREQ, frequency)
    chest_ecg = resample_frequency(chest_ecg, CHEST_FREQ, frequency)
    chest_emg = resample_frequency(chest_emg, CHEST_FREQ, frequency)
    chest_eda = resample_frequency(chest_eda, CHEST_FREQ, frequency)
    chest_temp = resample_frequency(chest_temp, CHEST_FREQ, frequency)
    chest_resp = resample_frequency(chest_resp, CHEST_FREQ, frequency)

    # Apply Butterworth filter
    chest_acc_x = butter_lowpass_filter(chest_acc_x, cutoff, frequency)
    chest_acc_y = butter_lowpass_filter(chest_acc_y, cutoff, frequency)
    chest_acc_z = butter_lowpass_filter(chest_acc_z, cutoff, frequency)
    chest_ecg = butter_lowpass_filter(chest_ecg, cutoff, frequency)
    chest_emg = butter_lowpass_filter(chest_emg, cutoff, frequency)
    chest_eda = butter_lowpass_filter(chest_eda, cutoff, frequency)
    chest_temp = butter_lowpass_filter(chest_temp, cutoff, frequency)
    chest_resp = butter_lowpass_filter(chest_resp, cutoff, frequency)

    timestamp = np.linspace(0, len(chest_acc_x) / frequency, len(chest_acc_x))

    df = pd.DataFrame({
        'chest_acc_x': chest_acc_x,
        'chest_acc_y': chest_acc_y,
        'chest_acc_z': chest_acc_z,
        'chest_ecg': chest_ecg,
        'chest_emg': chest_emg,
        'chest_eda': chest_eda,
        'chest_temp': chest_temp,
        'chest_resp': chest_resp,
        'timestamp': timestamp
    })

    return df


def load_subject(selected_subject, dataset_folder):
    print(f'Loading data for subject {selected_subject}')
    file_path = f'{dataset_folder}/{selected_subject}/{selected_subject}.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data


def apply_smote_to_subjects(train_data):
    smote = SMOTETomek()
    resampled_data_dict = {}
    for key, df in train_data.items():
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        try:
            X_resampled, y_resampled = smote.fit_resample(X, y)
        except ValueError:
            print(f'Skipping subject {key} due to resampling error')
            continue

        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled['label'] = y_resampled
        resampled_data_dict[key] = df_resampled

    return resampled_data_dict


def process_wrist_data_no_resample(data):
    print('Processing wrist data without resampling')
    wrist_acc_x = data['signal']['wrist']['ACC'][:, 0]
    wrist_acc_y = data['signal']['wrist']['ACC'][:, 1]
    wrist_acc_z = data['signal']['wrist']['ACC'][:, 2]
    wrist_bvp = data['signal']['wrist']['BVP'].flatten()
    wrist_eda = data['signal']['wrist']['EDA'].flatten()
    wrist_temp = data['signal']['wrist']['TEMP'].flatten()

    df_acc = pd.DataFrame({
        'wrist_acc_x': wrist_acc_x,
        'wrist_acc_y': wrist_acc_y,
        'wrist_acc_z': wrist_acc_z,
    })

    df_bvp = pd.DataFrame({'wrist_bvp': wrist_bvp})
    df_eda = pd.DataFrame({'wrist_eda': wrist_eda})
    df_temp = pd.DataFrame({'wrist_temp': wrist_temp})

    return df_acc, df_bvp, df_eda, df_temp


def process_chest_data_no_resample(data):
    print('Processing chest data without resampling')
    chest_acc_x = data['signal']['chest']['ACC'][:, 0]
    chest_acc_y = data['signal']['chest']['ACC'][:, 1]
    chest_acc_z = data['signal']['chest']['ACC'][:, 2]
    chest_ecg = data['signal']['chest']['ECG'].flatten()
    chest_emg = data['signal']['chest']['EMG'].flatten()
    chest_eda = data['signal']['chest']['EDA'].flatten()
    chest_temp = data['signal']['chest']['Temp'].flatten()
    chest_resp = data['signal']['chest']['Resp'].flatten()

    timestamp = np.linspace(0, len(chest_acc_x) / CHEST_FREQ, len(chest_acc_x))

    df = pd.DataFrame({
        'chest_acc_x': chest_acc_x,
        'chest_acc_y': chest_acc_y,
        'chest_acc_z': chest_acc_z,
        'chest_ecg': chest_ecg,
        'chest_emg': chest_emg,
        'chest_eda': chest_eda,
        'chest_temp': chest_temp,
        'chest_resp': chest_resp,
        'timestamp': timestamp
    })

    return df


def process_subjects(frequency, labels, window_size, preprocess_output_folder='data/WESAD/Preprocessed_Subjects',
                     dataset_folder='data/WESAD', subjects=SUBJECTS, resample_needed=True):
    for subject in subjects:
        try:
            print(f'Processing subject {subject}')
            data = load_subject(subject, dataset_folder)
            if resample_needed:
                wrist_df = process_wrist_data(data, frequency)
                chest_df = process_chest_data(data, frequency)
                df = pd.concat([wrist_df, chest_df], axis=1)
            else:
                wrist_df_acc, wrist_df_bvp, wrist_df_eda, wrist_df_temp = process_wrist_data_no_resample(data)
                chest_df = process_chest_data_no_resample(data)
                df = combine_wrist_chest_data_no_resample(wrist_df_acc, wrist_df_bvp, wrist_df_eda, wrist_df_temp,
                                                          chest_df)

            df = hampel_filter(df, window_size=window_size)
            labels_df = data['label']
            if resample_needed:
                labels_df = resample_labels(labels_df, CHEST_FREQ, frequency)
            df['label'] = labels_df
            df = convert_labels(df, target_labels=labels, binary=False)
            df = data_normalizer(df)
            os.makedirs(f'{preprocess_output_folder}/no_resampling', exist_ok=True)
            output_file = f'{preprocess_output_folder}/no_resampling/WESAD_{subject}.csv'
            df.to_csv(output_file, index=False)

            print(f'Output file for subject {subject}: {output_file}')
        except Exception as e:
            print(f'An error occurred while processing subject {subject}: {e}')


def load_all_subject_data(frequency, subjects=SUBJECTS, preprocess_output_folder='data/WESAD/Preprocessed_Subjects',
                          resample_needed=True):
    """Load all subjects"""
    all_dict = {}
    folder = 'resampling'
    for subject in subjects:
        df = pd.read_csv(f'{preprocess_output_folder}/{folder}/{frequency}/WESAD_{subject}_{frequency}.csv')
        all_dict[subject] = df
    return all_dict


def combine_wrist_chest_data_no_resample(wrist_df_acc, wrist_df_bvp, wrist_df_eda, wrist_df_temp, chest_df):
    print('Combining wrist and chest data without resampling')
    combined_df = pd.concat([chest_df, wrist_df_acc, wrist_df_bvp, wrist_df_eda, wrist_df_temp], axis=1)
    return combined_df
