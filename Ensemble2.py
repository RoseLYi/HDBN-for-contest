import argparse
import pickle
import numpy as np
from tqdm import tqdm
from skopt import gp_minimize

def load_scores(file_paths):
    """
    加载多个模型的预测得分，并返回一个列表。
    """
    scores = []
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            scores.append(list(pickle.load(f).items()))
    return scores

def objective(weights, scores, labels):
    """
    目标函数，用于评估给定权重下的准确率。
    """
    right_num = total_num = 0
    for i in tqdm(range(len(labels)), desc="Evaluating accuracy"):
        l = labels[i]
        combined_score = np.zeros_like(scores[0][i][1])
        for j, (score_list, weight) in enumerate(zip(scores, weights)):
            _, score = score_list[i]
            combined_score += score * weight
        prediction = np.argmax(combined_score)
        right_num += int(prediction == int(l))
        total_num += 1
    acc = right_num / total_num
    print(f"Accuracy: {acc:.4f}")
    return -acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize model ensemble weights using Gaussian Process.")
    
    # 添加命令行参数
    parser.add_argument('--benchmark', default='V1', choices=['V1', 'V2'], help="Benchmark version")
    parser.add_argument('--mixformer_J_Score', default='./Model_inference/Mix_Former/output/skmixf__V1_J/epoch1_test_score.pkl')
    parser.add_argument('--mixformer_B_Score', default='./Model_inference/Mix_Former/output/skmixf__V1_B/epoch1_test_score.pkl')
    parser.add_argument('--mixformer_JM_Score', default='./Model_inference/Mix_Former/output/skmixf__V1_JM/epoch1_test_score.pkl')
    parser.add_argument('--mixformer_BM_Score', default='./Model_inference/Mix_Former/output/skmixf__V1_BM/epoch1_test_score.pkl')
    parser.add_argument('--mixformer_k2_Score', default='./Model_inference/Mix_Former/output/skmixf__V1_k2/epoch1_test_score.pkl')
    parser.add_argument('--mixformer_k2M_Score', default='./Model_inference/Mix_Former/output/skmixf__V1_k2M/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_J2d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_V1_J/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_B2d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_JM2d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_V1_JM/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_BM2d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_V1_BM/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_J3d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_B3d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_JM3d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_V1_JM_3D/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_BM3d_Score', default='./Model_inference/Mix_GCN/output/ctrgcn_V1_BM_3D/epoch1_test_score.pkl')
    parser.add_argument('--tdgcn_J2d_Score', default='./Model_inference/Mix_GCN/output/tdgcn_V1_J/epoch1_test_score.pkl')
    parser.add_argument('--tdgcn_B2d_Score', default='./Model_inference/Mix_GCN/output/tdgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument('--tdgcn_JM2d_Score', default='./Model_inference/Mix_GCN/output/tdgcn_V1_JM/epoch1_test_score.pkl')
    parser.add_argument('--tdgcn_BM2d_Score', default='./Model_inference/Mix_GCN/output/tdgcn_V1_BM/epoch1_test_score.pkl')
    parser.add_argument('--mstgcn_J2d_Score', default='./Model_inference/Mix_GCN/output/mstgcn_V1_J/epoch1_test_score.pkl')
    parser.add_argument('--mstgcn_B2d_Score', default='./Model_inference/Mix_GCN/output/mstgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument('--mstgcn_JM2d_Score', default='./Model_inference/Mix_GCN/output/mstgcn_V1_JM/epoch1_test_score.pkl')
    parser.add_argument('--mstgcn_BM2d_Score', default='./Model_inference/Mix_GCN/output/mstgcn_V1_BM/epoch1_test_score.pkl')
    
    args = parser.parse_args()

    # 选择基准版本
    benchmark = args.benchmark
    npz_data = np.load(f'./Model_inference/Mix_Former/dataset/save_2d_pose/{benchmark}.npz')
    labels = npz_data['y_test']

    # 收集所有模型的预测得分文件路径
    file_paths = [
        args.mixformer_J_Score, args.mixformer_B_Score, args.mixformer_JM_Score, args.mixformer_BM_Score,
        args.mixformer_k2_Score, args.mixformer_k2M_Score, args.ctrgcn_J2d_Score, args.ctrgcn_B2d_Score,
        args.ctrgcn_JM2d_Score, args.ctrgcn_BM2d_Score, args.ctrgcn_J3d_Score, args.ctrgcn_B3d_Score,
        args.ctrgcn_JM3d_Score, args.ctrgcn_BM3d_Score, args.tdgcn_J2d_Score, args.tdgcn_B2d_Score,
        args.tdgcn_JM2d_Score, args.tdgcn_BM2d_Score, args.mstgcn_J2d_Score, args.mstgcn_B2d_Score,
        args.mstgcn_JM2d_Score, args.mstgcn_BM2d_Score
    ]

    # 加载模型预测得分
    scores = load_scores(file_paths)

    # 定义搜索空间
    search_space = [(0.2, 1.2) for _ in range(len(file_paths))]

    # 使用高斯过程优化权重
    result = gp_minimize(lambda w: objective(w, scores, labels), search_space, n_calls=200, random_state=0)

    # 输出最优结果
    print(f'Maximum accuracy: {result.fun * -100:.4f}%')
    print(f'Optimal weights: {result.x}')
