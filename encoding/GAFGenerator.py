import math
import numpy as np
from pyts.image import GramianAngularField
import torch
from torch.utils.data import Dataset

class GAFDataset(Dataset):
    def __init__(self, subjects, all_subject_data, features, labels, image_size=128, method='difference',
                 window_size=60 * 35, time_step=35, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.all_subject_data = all_subject_data
        self.features = features
        self.window_size = window_size
        self.gaf = GramianAngularField(image_size=image_size, sample_range=None, method=method)
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

    def get_image(self, subject, i):
        current_window = self.all_subject_data[subject].loc[i:i + self.window_size - 1, self.features].to_numpy().T
        gaf_image = self.gaf.transform(current_window)
        gaf_image = np.transpose(gaf_image, (1, 2, 0))
        return gaf_image

    def __getitem__(self, idx):
        subject, i, label = self.idx_list[idx]
        image = self.get_image(subject, i)
        y = label - 1
        return torch.tensor(image, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
