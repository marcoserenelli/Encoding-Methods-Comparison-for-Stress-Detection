import warnings
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from pyts.image import MarkovTransitionField
import torch
from torch.utils.data import Dataset

# Filter specific warnings
warnings.filterwarnings("ignore", message="Some quantiles are equal. The number of bins will be smaller for sample")

class MTFDataset(Dataset):
    def __init__(self, subjects, all_subject_data, features, labels, image_size=128, n_bins=5,
                 window_size=60 * 35, time_step=35, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.all_subject_data = all_subject_data
        self.features = features
        self.window_size = window_size
        self.n_bins = n_bins
        self.image_size = image_size
        self.mtf = MarkovTransitionField(image_size=image_size, n_bins=n_bins)
        self.idx_list = []

        for subject in subjects:
            if subject not in all_subject_data:
                print(f"Subject {subject} data is missing in all_subject_data.")
                continue
            for label in labels:
                curr_idx_interval = all_subject_data[subject].index[
                    all_subject_data[subject]['label'] == label].tolist()
                for i in range(curr_idx_interval[0], curr_idx_interval[-1] - window_size + 1, time_step):
                    self.idx_list.append((subject, i, label))

        self.shuffle_data()

    def shuffle_data(self):
        np.random.shuffle(self.idx_list)

    def __len__(self):
        return len(self.idx_list)

    def add_noise(self, data, noise_level=1e-5):
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise

    def get_image(self, subject, i):
        current_window = self.all_subject_data[subject].loc[i:i + self.window_size - 1, self.features].to_numpy().T

        # Apply noise to data
        current_window_noisy = self.add_noise(current_window)  # Use default noise level

        # Apply KBinsDiscretizer
        try:
            kbins = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                current_window_binned = kbins.fit_transform(current_window_noisy)
        except ValueError as e:
            print(f"ValueError for subject {subject}, index {i}: {e}")
            print(f"Current window shape: {current_window.shape}")
            print(f"Current window data: {current_window}")
            raise

        return np.transpose(self.mtf.transform(current_window_binned), (1, 2, 0))

    def __getitem__(self, idx):
        subject, i, label = self.idx_list[idx]
        image = self.get_image(subject, i)
        y = label - 1
        return torch.tensor(image, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
