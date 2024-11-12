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
    
    # 添加多个模型的预测得分文件路径
    parser.add_argument('--mixformer_J_Score', type=str, default='./Model_inference/Mix_GCN/output/ctrgcn_V1_J/epoch1_test_score.pkl')
    parser.add_argument('--mixformer_B_Score', type=str, default='./Model_inference/Mix_GCN/output/ctrgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument('--mixformer_JM_Score', type=str, default='./Model_inference/Mix_GCN/output/ctrgcn_V1_JM/epoch1_test_score.pkl')
    parser.add_argument('--mixformer_BM_Score', type=str, default='./Model_inference/Mix_GCN/output/ctrgcn_V1_BM/epoch1_test_score.pkl')
    parser.add_argument('--mixformer_k2_Score', type=str, default='./Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl')
    parser.add_argument('--mixformer_k2M_Score', type=str, default='./Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_J2d_Score', type=str, default='./Model_inference/Mix_GCN/output/ctrgcn_V1_J/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_B2d_Score', type=str, default='./Model_inference/Mix_GCN/output/ctrgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_JM2d_Score', type=str, default='./Model_inference/Mix_GCN/output/ctrgcn_V1_JM/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_BM2d_Score', type=str, default='./Model_inference/Mix_GCN/output/ctrgcn_V1_BM/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_J3d_Score', type=str, default='./Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_B3d_Score', type=str, default='./Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_JM3d_Score', type=str, default='./Model_inference/Mix_GCN/output/ctrgcn_V1_JM_3D/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_BM3d_Score', type=str, default='./Model_inference/Mix_GCN/output/ctrgcn_V1_BM_3D/epoch1_test_score.pkl')
    parser.add_argument('--tdgcn_J2d_Score', type=str, default='./Model_inference/Mix_GCN/output/tdgcn_V1_J/epoch1_test_score.pkl')
    parser.add_argument('--tdgcn_B2d_Score', type=str, default='./Model_inference/Mix_GCN/output/tdgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument('--tdgcn_JM2d_Score', type=str, default='./Model_inference/Mix_GCN/output/tdgcn_V1_JM/epoch1_test_score.pkl')
    parser.add_argument('--tdgcn_BM2d_Score', type=str, default='./Model_inference/Mix_GCN/output/tdgcn_V1_BM/epoch1_test_score.pkl')
    parser.add_argument('--mstgcn_J2d_Score', type=str, default='./Model_inference/Mix_GCN/output/mstgcn_V1_J/epoch1_test_score.pkl')
    parser.add_argument('--mstgcn_B2d_Score', type=str, default='./Model_inference/Mix_GCN/output/mstgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument('--mstgcn_JM2d_Score', type=str, default='./Model_inference/Mix_GCN/output/mstgcn_V1_JM/epoch1_test_score.pkl')
    parser.add_argument('--mstgcn_BM2d_Score', type=str, default='./Model_inference/Mix_GCN/output/mstgcn_V1_BM/epoch1_test_score.pkl')

    # 验证集样本文件路径
    parser.add_argument('--val_sample', type=str, default='./Process_data/CS_test_V1.txt')
    
    # 指定数据集基准版本
    parser.add_argument('--benchmark', type=str, default='V1')
    
    return parser

def load_scores(files):
    """
    加载多个模型的预测得分，并返回一个列表。
    """
    scores = []
    for file in files:
        with open(file, 'rb') as f:
            score = torch.tensor(pickle.load(f), dtype=torch.float32)
            scores.append(score)
    return scores

def calculate_weighted_score(scores, rates):
    """
    根据给定的权重计算加权后的综合得分。
    """
    weighted_scores = [score * rate for score, rate in zip(scores, rates)]
    final_score = torch.sum(torch.stack(weighted_scores), dim=0)
    return final_score

def calculate_accuracy(final_score, true_labels):
    """
    计算预测准确率。
    """
    _, predicted_labels = torch.max(final_score, 1)
    correct_predictions = (predicted_labels == true_labels).sum().item()
    accuracy = correct_predictions / len(true_labels)
    return accuracy

def generate_true_labels(val_txt_path):
    """
    从验证集样本文件中提取真实标签。
    """
    val_txt = np.loadtxt(val_txt_path, dtype=str)
    true_labels = [int(name.split('A')[1][:3]) for name in val_txt]
    return torch.tensor(true_labels, dtype=torch.long)

def main():
    """
    主函数，执行多模型融合及准确率计算。
    """
    parser = get_parser()
    args = parser.parse_args()

    # 收集所有模型的预测得分文件
    files = [
        args.mixformer_J_Score, args.mixformer_B_Score, args.mixformer_JM_Score, args.mixformer_BM_Score,
        args.mixformer_k2_Score, args.mixformer_k2M_Score, args.ctrgcn_J2d_Score, args.ctrgcn_B2d_Score,
        args.ctrgcn_JM2d_Score, args.ctrgcn_BM2d_Score, args.ctrgcn_J3d_Score, args.ctrgcn_B3d_Score,
        args.ctrgcn_JM3d_Score, args.ctrgcn_BM3d_Score, args.tdgcn_J2d_Score, args.tdgcn_B2d_Score,
        args.tdgcn_JM2d_Score, args.tdgcn_BM2d_Score, args.mstgcn_J2d_Score, args.mstgcn_B2d_Score,
        args.mstgcn_JM2d_Score, args.mstgcn_BM2d_Score
    ]
    
    # 根据基准版本选择相应的权重和样本数量
    if args.benchmark == 'V1':
        num_classes = 155
        sample_num = 6307
        rates = [1.2, 0.7687800313360855, 0.2, 0.2, 1.2, 1.2, 0.8474862468452808, 1.2, 0.2, 0.2, 0.6721599889400984, 0.8671683827594867, 0.2, 0.2, 0.7934336157353554, 1.2, 0.2, 0.2, 1.2, 1.2, 0.2, 0.2]
    elif args.benchmark == 'V2':
        num_classes = 155
        sample_num = 6599
        rates = [0.7214280414594167, 1.2, 0.2, 1.2, 1.2, 0.9495413913063555, 1.2, 1.2, 0.2, 0.2, 1.2, 1.2, 0.2, 0.2, 1.2, 1.2, 0.2, 0.2, 0.6745433985952421, 0.3926448734729191, 0.2, 0.2]
    else:
        raise ValueError("Unsupported benchmark version")

    # 加载模型预测得分
    scores = load_scores(files)
    
    # 计算加权后的综合得分
    final_score = calculate_weighted_score(scores, rates)
    
    # 生成真实标签
    true_labels = generate_true_labels(args.val_sample)
    
    # 计算并打印准确率
    accuracy = calculate_accuracy(final_score, true_labels)
    print(f'Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    main()
