import numpy as np
import os
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from feeders import tools
from .feeder_uav import Feeder

class UAV3DResampleRotateFeeder(Feeder):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_opts=None,
                 window_size=-1, normalization=False, debug=False, use_mmap=False, bone=False, vel=False):
        """
        UAV Human 3D Feeder with resampling and rotation options.
        :param data_path: Path to the data file.
        :param label_path: Path to the label file.
        :param split: 'train' or 'test' data split.
        :param random_opts: Dictionary for random data augmentations.
        :param window_size: Output sequence length.
        :param normalization: Apply normalization if True.
        :param debug: Use only the first 100 samples if True.
        :param use_mmap: Use memory mapping if True.
        :param bone: Enable bone modality.
        :param vel: Enable velocity modality.
        """
        super().__init__(data_path, label_path, p_interval, split, **(random_opts or {}), window_size, 
                         normalization, debug, use_mmap, bone, vel)
        self._load_data()
        if normalization:
            self.get_mean_map()

    def _load_data_from_file(self, file_path):
        with open(file_path, 'r') as f:
            data = [int(line.split(':')[1].strip()) for line in f]
        return np.array(data)

    def resample(self, data, labels):
        threshold = 105
        samples_table = self._load_data_from_file(os.path.join(os.getcwd(), "feeders/classes_samples.txt"))
        resampled_data, resampled_labels = [], []

        for class_idx, sample_count in enumerate(samples_table):
            indices = np.where(labels == class_idx)[0]
            class_samples = data[indices]
            extra_samples_needed = threshold - sample_count

            if extra_samples_needed > 0:
                extra_indices = np.random.choice(len(class_samples), extra_samples_needed, replace=True)
                extra_samples = class_samples[extra_indices]
                class_samples = np.vstack((class_samples, extra_samples))
            
            resampled_data.append(class_samples)
            resampled_labels.append(np.full(len(class_samples), class_idx))

        X, y = shuffle(resampled_data, resampled_labels, random_state=42)
        return np.concatenate(X), np.concatenate(y)

    def _load_data(self):
        data, labels = np.load(self.data_path), np.load(self.label_path)
        if self.split == 'train':
            data, labels = self.resample(data, labels)
        
        if self.debug:
            data, labels = data[:100], labels[:100]

        self.data, self.label = data, labels
        self.sample_name = [f"{self.split}_{i}" for i in range(len(self.data))]

    def __getitem__(self, index):
        data_sample = self.data[index]
        label = self.label[index]

        if not np.any(data_sample):
            data_sample = np.zeros((2, 64, 17, 300))

        valid_frames = np.sum(data_sample.sum(axis=(0, -1, -2)) != 0)
        data_sample = tools.valid_crop_resize(data_sample, valid_frames, self.p_interval, self.window_size)

        if self.random_rot:
            data_sample = tools.random_rot(data_sample, channel=3)
        if self.bone:
            data_sample = self._compute_bone_data(data_sample)
        if self.vel:
            data_sample[:, :-1] = data_sample[:, 1:] - data_sample[:, :-1]
            data_sample[:, -1] = 0

        return data_sample, label, index

    def _compute_bone_data(self, data):
        from .bone_pairs import ntu_pairs
        bone_data = np.zeros_like(data)
        for v1, v2 in ntu_pairs:
            bone_data[:, :, v1 - 1] = data[:, :, v1 - 1] - data[:, :, v2 - 1]
        return bone_data
