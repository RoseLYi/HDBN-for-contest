import numpy as np
from .feeder import Feeder
from . import tools

COCO_PAIRS = [
    (1, 6), (2, 1), (3, 1), (4, 2), (5, 3), (6, 7), (7, 1),
    (8, 6), (9, 7), (10, 8), (11, 9), (12, 6), (13, 7),
    (14, 12), (15, 13), (16, 14), (17, 15)
]

class FeederUAVHuman(Feeder):
    def __init__(self, data_path: str, label_path: str, data_split: str, 
                 p_interval: list = [0.95], window_size: int = 64, 
                 bone: bool = False, vel: bool = False, debug: bool = False):
        super().__init__(data_path, data_split, p_interval, window_size, bone, vel)
        self.label_path = label_path
        self.debug = debug
        self._load_data()

    def _load_data(self):
        """加载数据并设置样本名称。"""
        data = np.load(self.data_path)
        label = np.load(self.label_path)
        data_prefix = 'test_' if self.data_split == 'test' else 'train_'

        # Debug 模式下只加载前 100 条数据
        if self.debug:
            self.data, self.label = data[:100], label[:100]
            self.sample_name = [f"{data_prefix}{i}" for i in range(100)]
        else:
            self.data, self.label = data, label
            self.sample_name = [f"{data_prefix}{i}" for i in range(len(data))]
