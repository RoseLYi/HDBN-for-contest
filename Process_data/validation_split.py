import numpy as np
from sklearn.model_selection import train_test_split

# 文件路径
input_path = './data/train_joint.npy'
label_path = './data/train_label.npy'

# 加载数据
data = np.load(input_path)          # 输入数据形状 (N, C, T, V, M)
labels = np.load(label_path)         # 标签数据形状 (N, 155)

# 划分训练集和验证集
train_data, val_data, train_labels, val_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# 保存划分结果
np.save('train_data.npy', train_data)
np.save('val_data.npy', val_data)
np.save('train_labels.npy', train_labels)
np.save('val_labels.npy', val_labels)

print("数据集已成功划分并保存为 .npy 文件。")
