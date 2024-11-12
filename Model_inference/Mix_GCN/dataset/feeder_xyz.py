import torch
import numpy as np
from torch.utils.data import Dataset
from . import tools

# 定义关节对
COCO_PAIRS = [
    (1, 6), (2, 1), (3, 1), (4, 2), (5, 3), (6, 7), (7, 1), 
    (8, 6), (9, 7), (10, 8), (11, 9), (12, 6), (13, 7), 
    (14, 12), (15, 13), (16, 14), (17, 15)
]

class Feeder(Dataset):
    def __init__(self, data_path: str, data_split: str, p_interval: list = [0.95], 
                 window_size: int = 64, bone: bool = False, vel: bool = False):
        super().__init__()
        self.data_path = data_path
        self.data_split = data_split
        self.p_interval = p_interval
        self.window_size = window_size
        self.bone = bone
        self.vel = vel
        self.load_data()

    def load_data(self):
        """加载数据集和标签"""
        npz_data = np.load(self.data_path, allow_pickle=True)
        if self.data_split == 'train':
            self.data = npz_data['x_train']
            self.label = npz_data['y_train']
            prefix = 'train_'
        else:
            self.data = npz_data['x_test']
            self.label = npz_data['y_test']
            prefix = 'test_'
        self.sample_name = [f"{prefix}{i}" for i in range(len(self.data))]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        """获取一个样本及其标签"""
        data = self.data[idx]
        label = self.label[idx]
        data = np.array(data)
        
        # 计算有效帧数，避免全0帧
        valid_frame_num = np.sum(data.sum(0).sum(-1).sum(-1) != 0)
        if valid_frame_num == 0:
            return np.zeros((3, self.window_size, 17, 2)), label, idx
        
        # 数据预处理：裁剪并调整骨架数据
        data = tools.valid_crop_resize(data, valid_frame_num, self.p_interval, self.window_size)
        
        # 使用骨架模式
        if self.bone:
            bone_data = np.zeros_like(data)
            for v1, v2 in COCO_PAIRS:
                bone_data[:, :, v1 - 1] = data[:, :, v1 - 1] - data[:, :, v2 - 1]
            data = bone_data

        # 使用速度模式
        if self.vel:
            data[:, :-1] = data[:, 1:] - data[:, :-1]
            data[:, -1] = 0

        # 对齐所有关节点到第一个关节点
        data -= np.tile(data[:, :, 0:1, :], (1, 1, 17, 1))
        
        return data, label, idx

    def top_k(self, score, top_k):
        """计算 Top-K 准确率"""
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) / len(hit_top_k)
