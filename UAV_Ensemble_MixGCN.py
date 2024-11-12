import torch
import pickle
import argparse
import numpy as np
import pandas as pd

def get_parser():
    """
    创建命令行参数解析器，用于接收模型预测得分文件路径和其他配置信息。
    """
    parser = argparse.ArgumentParser(description='Multi-stream ensemble for action recognition models.')
    parser.add_argument(
        '--ctrgcn_J3d_Score', 
        type=str,
        default='./Model_inference/Mix_GCN/output/uav_human/ctrgcn_V1_J_3D/epoch1_test_score.pkl',
        help='Path to the CTRGCN J3D score file.'
    )
    parser.add_argument(
        '--ctrgcn_JBM3d_Score', 
        type=str,
        default='./Model_inference/Mix_GCN/output/uav_human/ctrgcn_V1_J_3D_bone_vel/epoch1_test_score.pkl',
        help='Path to the CTRGCN JBM3D score file.'
    )
    parser.add_argument(
        '--val_sample', 
        type=str,
        default='./Process_data/CS_test_V1.txt',
        help='Path to the validation sample file.'
    )
    parser.add_argument(
        '--benchmark', 
        type=str,
        default='V1',
        choices=['V1', 'V2'],
        help='Benchmark version (V1 or V2).'
    )
    return parser

def cal_score(file_paths, rates, num_samples, num_classes):
    """
    计算加权后的综合得分。
    """
    final_score = torch.zeros(num_samples, num_classes)
    for idx, file_path in enumerate(file_paths):
        with open(file_path, 'rb') as f:
            inf = pickle.load(f)
            score = torch.tensor(inf, dtype=torch.float32)
            final_score += rates[idx] * score
    return final_score

def cal_acc(final_score, true_label):
    """
    计算预测准确率。
    """
    wrong_indices = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label != true_label[index]:
            wrong_indices.append(index)
    
    wrong_num = len(wrong_indices)
    print(f'Wrong predictions: {wrong_num}')
    
    total_num = true_label.size(0)
    print(f'Total samples: {total_num}')
    acc = (total_num - wrong_num) / total_num
    return acc

def gen_label(val_txt_path):
    true_label = []
    val_txt = np.loadtxt(val_txt_path, dtype=str)
    for name in val_txt:
        label = int(name.split('A')[1][:3])
        true_label.append(label)
    
    true_label = torch.tensor(true_label, dtype=torch.long)
    return true_label

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # 模型预测得分文件路径
    j_file = args.ctrgcn_J3d_Score
    b_file = args.ctrgcn_JBM3d_Score
    val_txt_file = args.val_sample

    file_paths = [j_file, b_file]
    
    if args.benchmark == 'V1':
        num_classes = 155
        num_samples = 2000
        rates = [0.6, 0.4]
    elif args.benchmark == 'V2':
        num_classes = 155
        num_samples = 6599
        rates = [0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.05, 0.05, 0.05, 0.05]
    else:
        raise ValueError("Unsupported benchmark version")

    final_score = cal_score(file_paths, rates, num_samples, num_classes)
    true_label = gen_label(val_txt_file)
    acc = cal_acc(final_score, true_label)
    print(f'Accuracy: {acc:.4f}')
