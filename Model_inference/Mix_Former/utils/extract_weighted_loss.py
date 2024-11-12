import numpy as np

def compute_class_weights(labels):
    """
    计算每个类别的权重，基于样本分布。
    :param labels: np.ndarray, 样本标签数组
    :return: np.ndarray, 每个类别的权重
    """
    num_classes = 155
    sample_counts = np.bincount(labels, minlength=num_classes)  # 计算每个类的样本数
    print(f"Sample counts per class: {sample_counts}")

    # 避免除零，初始化权重
    weights = np.zeros(num_classes, dtype=np.float32)
    for idx, count in enumerate(sample_counts):
        weights[idx] = labels.size / count if count > 0 else 0  # 计算权重，避免零除错误

    print(f"Class weights: {weights}")
    return weights
