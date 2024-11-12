import pickle
import numpy as np

def load_pickle_file(file_path):
    """加载并返回 pickle 文件的数据"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_npy_file(file_path):
    """加载并返回 numpy 文件的数据"""
    with open(file_path, 'rb') as f:
        data = np.load(f)
    return data

def main():
    # 加载并打印 Mix_Former 模型的得分
    mixformer_score = load_pickle_file('./Mix_Former/mixformer_J.pkl')
    print(mixformer_score.get("test_0"))

    # 加载并打印测试标签的维度
    test_labels = load_npy_file('./test_label_A.npy')
    print(test_labels.shape)

if __name__ == "__main__":
    main()
