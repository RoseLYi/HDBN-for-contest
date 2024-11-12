import numpy as np
import torch
from torch.utils.data import Dataset
from .feeder_xyz import Feeder
from . import tools

# 关节连接关系
COCO_PAIRS = [
    (1, 6), (2, 1), (3, 1), (4, 2), (5, 3), (6, 7), (7, 1), 
    (8, 6), (9, 7), (10, 8), (11, 9), (12, 6), (13, 7), 
    (14, 12), (15, 13), (16, 14), (17, 15)
]

class FeederUAVHuman(Feeder):
    def __init__(self, data_path: str, label_path: str, data_split: str, 
                 p_interval: list = [0.95], window_size: int = 64,
                 bone: bool = False, vel: bool = False, debug: bool = False):
        """初始化数据加载器"""
        super().__init__(data_path, data_split, p_interval, window_size, bone, vel)
        self.label_path = label_path
        self.debug = debug
        self._load_data()

    def _load_data(self):
        """加载数据和标签，根据数据集划分设置样本名称"""
        data = np.load(self.data_path)
        label = np.load(self.label_path)
        prefix = 'test_' if self.data_split == 'test' else 'train_'
        
        # 使用部分数据进行调试
        if self.debug:
            data, label = data[:100], label[:100]
        
        self.data = data
        self.label = label
        self.sample_name = [f"{prefix}{i}" for i in range(len(data))]
