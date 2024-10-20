from .feeder_xyz import Feeder;
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from . import tools;

coco_pairs = [(10, 8), (8, 6), (9, 7), (7, 5), 
                    (15, 13), (13, 11),(16, 14), (14, 12), 
                    (11, 5), (12, 6), (11, 12), (5, 6), 
                    (5, 0), (6, 0), (1, 0), (2, 0), 
                    (3, 1), (4, 2)]
class FeederUAVHuman(Feeder):
    #构造函数初始化
    def __init__(self, data_path: str, label_path: str,   #这里改成label_path, 因为我们比赛的数据集和标签分开的
                 data_split: str, #还是加回去data_split 
                 p_interval: list=[0.95], window_size: int=64,
                 bone: bool=False, vel: bool=False,
                 debug=False,  # 添加一个debug好了, 便于调试我们的feeder正不正确 
                 ):  
        #初始化父类
        super().__init__(data_path, data_split, p_interval, window_size, bone, vel);
        #初始化自己的属性
        self.label_path = label_path;
        self.debug = debug;
        self.load_data();

    # 加载数据的方法
    def load_data(self):

        # 两个都加载, 按照不同阶段返回东西就好
        # data和label可以代表训练集的，也可以代表测试集的
        data = np.load(self.data_path);
        label = np.load(self.label_path);

        #TODO 重采样

        # 这里sample_name需要一点判断，还是写上好了
        data_type_name = 'test_' if self.data_split == 'test' else 'train_';

        # 刚刚大概看了一下，还是按照一起加载的逻辑来写比较好(先暂时这样), 不然其他地方也要改
        if not self.debug:
            self.data = data;
            self.label = label;
            self.sample_name = [data_type_name + str(i) for i in range(len(self.data))];  #还是给一个sample_name吧
        else:
            self.data = data[0:100];
            self.label = label[0:100];
            self.sample_name = [data_type_name + str(i) for i in range(100)];

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        data_numpy = self.data[idx] # C,T,V,M
        label = self.label[idx]
        #data_numpy = np.transpose(data_numpy, (3, 1, 2, 0)) # C,T,V,M
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        if(valid_frame_num == 0): 
            return np.zeros((3, 64, 17, 300)), label, idx
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in coco_pairs:
                bone_data_numpy[:, :, v1] = data_numpy[:, :, v1] - data_numpy[:, :, v2]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        data_numpy = data_numpy - np.tile(data_numpy[:, :, 0:1, :], (1, 1, 17, 1)) # all_joint - 0_joint
        return data_numpy, label, idx # C T V M