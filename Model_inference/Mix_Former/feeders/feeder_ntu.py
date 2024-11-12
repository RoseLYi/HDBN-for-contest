import numpy as np
from torch.utils.data import Dataset
from feeders import tools


class SkeletonFeeder(Dataset):
    def __init__(self, data_path, label_path=None, split='train', window_size=-1, 
                 normalization=False, debug=False, use_mmap=False, p_interval=1,
                 random_opts=None, bone=False, vel=False):
        """
        Skeleton dataset loader with optional data augmentation.
        :param data_path: Path to the data file.
        :param label_path: Path to the label file.
        :param split: Data split, 'train' or 'test'.
        :param window_size: Length of output sequence.
        :param normalization: If true, normalize data.
        :param debug: If true, loads only a subset of data.
        :param use_mmap: If true, load data with memory mapping.
        :param p_interval: Interval for data cropping.
        :param random_opts: Dictionary of random operations (choose, shift, move, rot).
        :param bone: Whether to use bone modality.
        :param vel: Whether to use velocity modality.
        """
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.window_size = window_size
        self.normalization = normalization
        self.debug = debug
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_opts = random_opts if random_opts else {}
        self.bone = bone
        self.vel = vel

        self.data, self.label = self._load_data()
        if normalization:
            self._compute_normalization_stats()

    def _load_data(self):
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            data, label = npz_data['x_train'], np.where(npz_data['y_train'] > 0)[1]
        elif self.split == 'test':
            data, label = npz_data['x_test'], np.where(npz_data['y_test'] > 0)[1]
        else:
            raise ValueError("split must be 'train' or 'test'")
        
        data = data.reshape(data.shape[0], data.shape[1], 2, 25, 3).transpose(0, 4, 1, 3, 2)
        return data, label

    def _compute_normalization_stats(self):
        N, C, T, V, M = self.data.shape
        self.mean_map = self.data.mean(axis=(2, 4), keepdims=True).mean(axis=0)
        self.std_map = self.data.transpose(0, 2, 4, 1, 3).reshape(N * T * M, C * V).std(axis=0).reshape(C, 1, V, 1)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sample_data = np.array(self.data[idx])
        label = self.label[idx]
        valid_frames = np.sum(sample_data.sum(axis=(0, -1, -2)) != 0)

        sample_data = tools.valid_crop_resize(sample_data, valid_frames, self.p_interval, self.window_size)
        if self.random_opts.get('rot'):
            sample_data = tools.random_rot(sample_data)
        if self.bone:
            sample_data = self._apply_bone_mode(sample_data)
        if self.vel:
            sample_data[:, :-1] = sample_data[:, 1:] - sample_data[:, :-1]
            sample_data[:, -1] = 0

        return sample_data, label, idx

    def _apply_bone_mode(self, data):
        from .bone_pairs import ntu_pairs
        bone_data = np.zeros_like(data)
        for v1, v2 in ntu_pairs:
            bone_data[:, :, v1 - 1] = data[:, :, v1 - 1] - data[:, :, v2 - 1]
        return bone_data

    def top_k_accuracy(self, scores, k):
        ranked_scores = scores.argsort()
        hits = [lbl in ranked_scores[i, -k:] for i, lbl in enumerate(self.label)]
        return sum(hits) / len(hits)


def dynamic_import(name):
    components = name.split('.')
    module = __import__(components[0])
    for comp in components[1:]:
        module = getattr(module, comp)
    return module
