import numpy as np
import torch
from torch.utils.data import Dataset
from recurrence_plot import create_multivariate_recurrence_plot, DistanceMethodName

class RPDataset(Dataset):
    def __init__(self, subjects, all_subject_data, features, labels, image_size=128, threshold=0.5,
                 window_size=60 * 35, time_step=35, shuffle=True, seed=None, distance='dtw'):
        if seed is not None:
            np.random.seed(seed)
        # Initialize attributes
        self.all_subject_data = all_subject_data
        self.features = features
        self.window_size = window_size
        self.image_size = image_size
        self.channels = len(features)
        self.shuffle = shuffle
        self.threshold = threshold
        self.distance = distance
        self.idx_list = []

        # Create index list for windows
        for subject in subjects:
            for label in labels:
                curr_idx_interval = all_subject_data[subject].index[
                    all_subject_data[subject]['label'] == label].tolist()
                for i in range(curr_idx_interval[0], curr_idx_interval[-1] - window_size + 1, time_step):
                    self.idx_list.append((subject, i, label))

        if self.shuffle:
            self.shuffle_data()

    def shuffle_data(self):
        np.random.shuffle(self.idx_list)

    def __len__(self):
        return len(self.idx_list)

    def get_image(self, subject, i):
        current_window = self.all_subject_data[subject].iloc[i:i + self.window_size, :][self.features]
        if self.distance == 'dtw':
            non_nan_series = [series.dropna().to_numpy() for series in current_window.T]
            # Create a multivariate array from these series
            multivariate_series = np.array([series[:self.window_size] for series in non_nan_series])
            rp_image = create_multivariate_recurrence_plot(multivariate_series, threshold=self.threshold, distance=self.distance,
                                                           image_size=self.image_size)
        else:
            current_window = current_window.dropna(axis=0, how='any').to_numpy().T
            rp_image = create_multivariate_recurrence_plot(current_window, threshold=self.threshold, distance=self.distance,
                                                           image_size=self.image_size)
        return rp_image

    def __getitem__(self, idx):
        subject, i, label = self.idx_list[idx]
        image = self.get_image(subject, i)
        y = label - 1
        return torch.tensor(image, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
