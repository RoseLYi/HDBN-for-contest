import numpy as np
import os
import torch
from torch.utils.data import Dataset
from .feeder_xyz import Feeder
from . import tools
from sklearn.utils import shuffle

COCO_PAIRS = [
    (1, 6), (2, 1), (3, 1), (4, 2), (5, 3), (6, 7), (7, 1),
    (8, 6), (9, 7), (10, 8), (11, 9), (12, 6), (13, 7), (14, 12),
    (15, 13), (16, 14), (17, 15)
]

class FeederUAVHumanResampleRotate(Feeder):
    def __init__(self, data_path: str, label_path: str, data_split: str, 
                 p_interval: list = [0.95], window_size: int = 64,
                 bone: bool = False, vel: bool = False, random_rotate: bool = False,
                 debug: bool = False):
        super().__init__(data_path, data_split, p_interval, window_size, bone, vel)
        self.label_path = label_path
        self.random_rotate = random_rotate
        self.debug = debug
        self.load_data()

    def load_data_from_txt(self, file_path):
        """从文本文件中加载数据并返回 NumPy 数组。"""
        with open(file_path, 'r') as file:
            data = [int(line.strip().split(':')[1]) for line in file]
        return np.array(data)

    def resample(self, data, labels, threshold=105):
        """对每类数据进行重采样以达到指定阈值。"""
        sample_counts = self.load_data_from_txt(os.path.join(os.getcwd(), "dataset/classes_samples.txt"))
        resampled_data, resampled_labels = [], []

        for class_id, count in enumerate(sample_counts):
            class_indices = np.where(labels == class_id)[0]
            class_samples = data[class_indices]

            if count < threshold:
                additional_samples = np.random.choice(class_samples, threshold - count, replace=True)
                class_samples = np.concatenate((class_samples, additional_samples), axis=0)

            resampled_data.append(class_samples)
            resampled_labels.append(np.full(len(class_samples), class_id))

        X, y = shuffle(resampled_data, resampled_labels, random_state=42)
        return np.concatenate(X), np.concatenate(y)

    def load_data(self):
        """加载数据集并处理为所需格式。"""
        data, label = np.load(self.data_path), np.load(self.label_path)
        
        if self.data_split == 'train':
            data, label = self.resample(data, label)
            print(f"data: {data.shape}, label: {label.shape}")

        prefix = 'test_' if self.data_split == 'test' else 'train_'
        self.data = data[:100] if self.debug else data
        self.label = label[:100] if self.debug else label
        self.sample_name = [f"{prefix}{i}" for i in range(len(self.data))]

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        label = self.label[idx]
        valid_frames = np.sum(sample.sum(axis=(0, -1, -1)) != 0)

        if valid_frames == 0:
            return torch.zeros((3, 64, 17, 2)), label, idx

        sample = tools.valid_crop_resize(sample, valid_frames, self.p_interval, self.window_size)

        if self.bone:
            bone_data = np.zeros_like(sample)
            for v1, v2 in COCO_PAIRS:
                bone_data[:, :, v1 - 1] = sample[:, :, v1 - 1] - sample[:, :, v2 - 1]
            sample = bone_data

        if self.vel:
            sample[:, :-1] = sample[:, 1:] - sample[:, :-1]
            sample[:, -1] = 0

        if self.random_rotate:
            sample = tools.random_rot(sample)

        sample = sample - sample[:, :, :1, :]
        return sample, label, idx
