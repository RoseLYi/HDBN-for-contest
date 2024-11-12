import numpy as np
from .feeder_uav import Feeder
from feeders import tools

class UAVHumanFeeder(Feeder):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_options=None, window_size=-1, 
                 normalization=False, debug=False, use_mmap=False, bone=False, vel=False):  
        """
        Initialize UAV Human Feeder with data loading and augmentation options.
        :param data_path: Path to the data file.
        :param label_path: Path to the label file.
        :param p_interval: Interval parameter.
        :param split: Data split, either 'train' or 'test'.
        :param random_options: Dictionary with options like 'choose', 'shift', 'move', 'rot'.
        :param window_size: Target sequence length.
        :param normalization: Apply normalization if True.
        :param debug: Use only first 100 samples for debugging if True.
        :param use_mmap: Enable memory mapping.
        :param bone: Enable bone modality.
        :param vel: Enable velocity modality.
        """
        super().__init__(data_path, label_path, p_interval, split, **(random_options or {}), window_size, 
                         normalization, debug, use_mmap, bone, vel)
        self._load_data()
        if normalization:
            self._compute_mean_std()

    def _load_data(self):
        """Load data and set up sample names based on split type and debugging mode."""
        data, labels = np.load(self.data_path), np.load(self.label_path)
        prefix = 'test_' if self.split == 'test' else 'train_'
        
        if self.debug:
            self.data, self.label = data[:100], labels[:100]
            self.sample_name = [f"{prefix}{i}" for i in range(100)]
        else:
            self.data, self.label = data, labels
            self.sample_name = [f"{prefix}{i}" for i in range(len(data))]

    def __getitem__(self, index):
        """Retrieve sample data, apply augmentations and return."""
        data_sample = np.array(self.data[index])
        label = self.label[index]

        if not np.any(data_sample):
            data_sample = np.array(self.data[0])  # Fallback to the first sample if empty

        valid_frames = np.sum(data_sample.sum(axis=(0, -1, -2)) != 0)
        if valid_frames == 0:
            data_sample = np.zeros((2, 64, 17, 300))

        # Resize data based on valid frames and apply optional transformations
        data_sample = tools.valid_crop_resize(data_sample, valid_frames, self.p_interval, self.window_size)
        data_sample = self._apply_transformations(data_sample)

        return data_sample, label, index

    def _apply_transformations(self, data_sample):
        """Apply random rotation, bone transformation, and velocity transformations as needed."""
        if self.random_rot:
            data_sample = tools.random_rot(data_sample, channel=2)
        if self.bone:
            data_sample = self._apply_bone_modality(data_sample)
        if self.vel:
            data_sample[:, :-1] = data_sample[:, 1:] - data_sample[:, :-1]
            data_sample[:, -1] = 0
        return data_sample

    def _apply_bone_modality(self, data):
        """Compute bone modality data based on predefined bone pairs."""
        from .bone_pairs import ntu_pairs
        bone_data = np.zeros_like(data)
        for v1, v2 in ntu_pairs:
            bone_data[:, :, v1 - 1] = data[:, :, v1 - 1] - data[:, :, v2 - 1]
        return bone_data
