import numpy as np
import os
import torch
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from .feeder_xyz import Feeder
from . import tools

COCO_PAIRS = [
    (10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 11), 
    (16, 14), (14, 12), (11, 5), (12, 6), (11, 12), (5, 6), 
    (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2)
]

class FeederUAVHumanResample(Feeder):
    def __init__(self, data_path: str, label_path: str, data_split: str, 
                 p_interval: list = [0.95], window_size: int = 64,
                 bone: bool = False, vel: bool = False, debug: bool = False):
        super().__init__(data_path, data_split, p_interval, window_size, bone, vel)
        self.label_path = label_path
        self.debug = debug
        self.load_data()

    def load_data_from_txt(self, file_path):
        """读取文本文件并返回数据数组。"""
        with open(file_path, 'r') as file:
            return np.array([int(line.split(':')[1].strip()) for line in file])

    def resample(self, data, labels, threshold=105):
        """调整类样本数量，使每类达到指定阈值。"""
        sample_counts = self.load_data_from_txt(os.path.join(os.getcwd(), "dataset/classes_samples.txt"))
        resampled_data, resampled_labels = [], []

        for class_id, count in enumerate(sample_counts):
            class_data = data[labels == class_id]
            if count < threshold:
                resample_indices = np.random.choice(class_data.shape[0], threshold - count, replace=True)
                class_data = np.concatenate([class_data, class_data[resample_indices]], axis=0)
            resampled_data.append(class_data)
            resampled_labels.append(np.full(class_data.shape[0], class_id))

        return np.concatenate(resampled_data), np.concatenate(resampled_labels)

    def load_data(self):
        """加载数据并根据是否为训练集进行重采样。"""
        data, labels = np.load(self.data_path), np.load(self.label_path)
        
        if self.data_split == 'train':
            data, labels = self.resample(data, labels)
            print(f"Resampled data shape: {data.shape}, labels shape: {labels.shape}")

        prefix = 'test_' if self.data_split == 'test' else 'train_'
        self.data = data[:100] if self.debug else data
        self.label = labels[:100] if self.debug else labels
        self.sample_name = [f"{prefix}{i}" for i in range(len(self.data))]
