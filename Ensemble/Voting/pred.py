import pickle
import numpy as np

# 模型路径字典
model_paths = {
    "gcn": [
        "../scores/Mix_GCN/test/ctrgcn_V1_JM_3d.pkl",
        "../scores/Mix_GCN/test/ctrgcn_V1_B_3d.pkl",
        "../scores/Mix_GCN/test/ctrgcn_V1_J_3d.pkl",
        "../scores/Mix_GCN/test/ctrgcn_V1_J_3d_resample.pkl",
        "../scores/Mix_GCN/test/ctrgcn_V1_J_3d_resample_rotate.pkl",
        "../scores/Mix_GCN/test/ctrgcn_V1_B_2d.pkl",
        "../scores/Mix_GCN/test/ctrgcn_V1_J_2d.pkl",
        "../scores/Mix_GCN/test/ctrgcn_V1_BM_2d.pkl",
        "../scores/Mix_GCN/test/ctrgcn_V1_JM_2d.pkl",
        "../scores/Mix_GCN/test/tdgcn_V1_J_2d.pkl",
        "../scores/Mix_GCN/test/blockgcn_J_3d.pkl",
        "../scores/Mix_GCN/test/blockgcn_JM_3d.pkl",
        "../scores/Mix_GCN/test/blockgcn_B_3d.pkl",
        "../scores/Mix_GCN/test/blockgcn_BM_3d.pkl",
        "../scores/Mix_GCN/test/ctrgcn_V1_B_3d_resample_rotate.pkl",
        "../scores/Mix_GCN/test/degcn_J_3d.pkl",
        "../scores/Mix_GCN/test/degcn_B_3d.pkl",
        "../scores/Mix_GCN/test/degcn_BM_3d.pkl",
        "../scores/Mix_GCN/test/tegcn_V1_J_3d.pkl",
        "../scores/Mix_GCN/test/tegcn_V1_B_3d.pkl"
    ],
    "former": [
        "../scores/Mix_Former/test/mixformer_BM_r_w_2d.pkl",
        "../scores/Mix_Former/test/mixformer_BM_2d.pkl",
        "../scores/Mix_Former/test/mixformer_J_2d.pkl",
        "../scores/Mix_Former/test/mixformer_J_3d.pkl",
        "../scores/Mix_Former/test/mixformer_B_3d.pkl",
        "../scores/Mix_Former/test/mixformer_J_3d_resample_rotate.pkl",
        "../scores/Mix_Former/test/mixformer_JM_2d.pkl",
        "../scores/Mix_Former/test/mixformer_B_3d_resample_rotate.pkl",
        "../scores/Mix_Former/test/skateformer_B_3d.pkl",
        "../scores/Mix_Former/test/skateformer_J_3d.pkl"
    ]
}

# 初始权重列表
weights = [
    1.911, -0.452, 2.033, -0.214, 1.072, -2.569, 0.133, -2.680, 2.933, 
    4.711, 4.112, -1.498, 4.447, 1.322, 3.245, 2.590, 3.413, 3.391, 
    2.693, 0.498, 1.999, 2.043, -0.593, -0.183, -1.605, 2.791, -0.490, 
    1.538, 3.749, 7.709
]

# 加载预处理数据
def load_data(gcn=False, former=False):
    data_list = []
    for model_type, paths in model_paths.items():
        if (model_type == "gcn" and gcn) or (model_type == "former" and former):
            for path in paths:
                with open(path, 'rb') as f:
                    data_dict = pickle.load(f)
                data = np.array([data_dict[f"test_{i}"] for i in range(4599)])
                data_list.append(data)
    X = np.transpose(data_list, (1, 0, 2))  # 转置为 (samples, models, features)
    return X

def softmax(X):
    return np.exp(X) / np.sum(np.exp(X), axis=0, keepdims=True)  # 逐列执行softmax

def compute_confidence(X, weights):
    # 计算每个样本的加权置信度
    return np.array([
        np.sum([weights[i] * softmax(X[sample_idx][i]) for i in range(X.shape[1])], axis=0)
        for sample_idx in range(X.shape[0])
    ])

if __name__ == "__main__":
    X = load_data(gcn=True, former=True)
    confidences = compute_confidence(X, weights)
    
    # 保存置信度到文件
    np.save("pred.npy", confidences)
    print("Confidence scores saved to 'pred.npy'")
