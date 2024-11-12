import torch
import pickle
import argparse
import numpy as np
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser(description="Multi-stream ensemble")
    parser.add_argument('--ctrgcn_J3d_Score', type=str, default='./Model_inference/Mix_GCN/output/uav_human/ctrgcn_V1_J_3D/epoch1_test_score.pkl')
    parser.add_argument('--ctrgcn_JBM3d_Score', type=str, default='./Model_inference/Mix_GCN/output/uav_human/ctrgcn_V1_J_3D_bone_vel/epoch1_test_score.pkl')
    parser.add_argument('--val_sample', type=str, default='./Process_data/CS_test_V1.txt')
    parser.add_argument('--benchmark', type=str, default='V1')
    return parser

def calculate_score(file_paths, weights, num_samples, num_classes):
    final_score = torch.zeros(num_samples, num_classes)
    for idx, file_path in enumerate(file_paths):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        score = torch.tensor(pd.DataFrame(data).T.values)
        final_score += weights[idx] * score
    return final_score

def calculate_accuracy(final_score, true_labels):
    _, pred_labels = torch.max(final_score, 1)
    accuracy = (pred_labels == true_labels).float().mean().item()
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

def generate_labels(file_path):
    labels = [int(name.split('A')[1][:3]) for name in np.loadtxt(file_path, dtype=str)]
    return torch.tensor(labels)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Configuration for each benchmark
    benchmark_configs = {
        'V1': {
            'num_classes': 155,
            'num_samples': 2000,
            'weights': [0.6, 0.4],
            'file_paths': [args.ctrgcn_J3d_Score, args.ctrgcn_JBM3d_Score]
        },
        'V2': {
            'num_classes': 155,
            'num_samples': 6599,
            'weights': [0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.05, 0.05, 0.05, 0.05],
            'file_paths': [args.ctrgcn_J3d_Score, args.ctrgcn_JBM3d_Score]  # Update as needed for V2
        }
    }

    # Get configuration based on benchmark
    config = benchmark_configs[args.benchmark]
    
    # Load scores and labels
    final_score = calculate_score(config['file_paths'], config['weights'], config['num_samples'], config['num_classes'])
    true_labels = generate_labels(args.val_sample)
    
    # Calculate and print accuracy
    calculate_accuracy(final_score, true_labels)
