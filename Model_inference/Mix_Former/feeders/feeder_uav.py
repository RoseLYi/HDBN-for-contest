import numpy as np
from torch.utils.data import Dataset
from feeders import tools


class SkeletonFeeder(Dataset):
    def __init__(self, data_path, label_path=None, interval=1, mode='train', augmentation=None, window_size=-1,
                 normalize=False, debug_mode=False, mmap=False, use_bone=False, use_velocity=False):
        """
        Initialize the feeder for skeleton data.
        :param data_path: Path to the data file.
        :param label_path: Path to the label file.
        :param mode: Dataset mode, either 'train' or 'test'.
        :param augmentation: Dictionary containing augmentation options (choose, shift, move, rot).
        :param window_size: Target length of sequence.
        :param normalize: Apply normalization if True.
        :param debug_mode: Use only first 100 samples for debugging if True.
        :param mmap: Use memory mapping if True.
        :param use_bone: Enable bone modality.
        :param use_velocity: Enable velocity modality.
        """
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.augmentation = augmentation or {}
        self.window_size = window_size
        self.normalize = normalize
        self.debug_mode = debug_mode
        self.mmap = mmap
        self.interval = interval
        self.use_bone = use_bone
        self.use_velocity = use_velocity

        self.data, self.label = self._load_data()
        if normalize:
            self.mean_map, self.std_map = self._calculate_mean_std()

    def _load_data(self):
        """Load data and set sample names based on mode."""
        loaded_data = np.load(self.data_path)
        if self.mode == 'train':
            data, labels = loaded_data['x_train'], loaded_data['y_train']
            sample_names = [f'train_{i}' for i in range(len(data))]
        elif self.mode == 'test':
            data, labels = loaded_data['x_test'], loaded_data['y_test']
            sample_names = [f'test_{i}' for i in range(len(data))]
        else:
            raise ValueError("Mode should be 'train' or 'test'")
        data = data.transpose(0, 4, 1, 3, 2)  # Rearrange data dimensions for consistency
        return data, labels

    def _calculate_mean_std(self):
        """Calculate mean and standard deviation maps for normalization."""
        N, C, T, V, M = self.data.shape
        mean_map = self.data.mean(axis=(2, 4), keepdims=True).mean(axis=0)
        std_map = self.data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
        return mean_map, std_map

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        """Retrieve a single sample, apply augmentations if necessary, and return."""
        sample = np.array(self.data[index])
        label = self.label[index]
        
        if not sample.any():
            sample = np.array(self.data[0])  # Fallback to the first sample if empty

        valid_frames = np.sum(sample.sum(axis=(0, -1, -2)) != 0)
        sample = tools.valid_crop_resize(sample, valid_frames, self.interval, self.window_size)
        
        if self.augmentation.get('random_rot'):
            sample = tools.random_rot(sample)
        if self.use_bone:
            sample = self._apply_bone_modality(sample)
        if self.use_velocity:
            sample[:, :-1] = sample[:, 1:] - sample[:, :-1]
            sample[:, -1] = 0

        return sample, label, index

    def _apply_bone_modality(self, sample):
        """Apply bone modality by calculating bone differences based on skeleton pairs."""
        from .bone_pairs import ntu_pairs
        bone_sample = np.zeros_like(sample)
        for v1, v2 in ntu_pairs:
            bone_sample[:, :, v1 - 1] = sample[:, :, v1 - 1] - sample[:, :, v2 - 1]
        return bone_sample

    def top_k_accuracy(self, scores, k):
        """Compute top-k accuracy for the dataset."""
        rank = scores.argsort()
        top_k_hits = sum([label in rank[i, -k:] for i, label in enumerate(self.label)])
        return top_k_hits / len(self.label)


def dynamic_import(module_name):
    """Dynamically import a module given its dotted path."""
    components = module_name.split('.')
    module = __import__(components[0])
    for comp in components[1:]:
        module = getattr(module, comp)
    return module
