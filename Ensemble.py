import torch
import pickle
import argparse
import numpy as np
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser(description='Multi-stream ensemble')
    
    score_files = {
        'mixformer_J': './Model_inference/Mix_GCN/output/ctrgcn_V1_J/epoch1_test_score.pkl',
        'mixformer_B': './Model_inference/Mix_GCN/output/ctrgcn_V1_B/epoch1_test_score.pkl',
        'mixformer_JM': './Model_inference/Mix_GCN/output/ctrgcn_V1_JM/epoch1_test_score.pkl',
        'mixformer_BM': './Model_inference/Mix_GCN/output/ctrgcn_V1_BM/epoch1_test_score.pkl',
        'mixformer_k2': './Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl',
        'mixformer_k2M': './Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl',
        'ctrgcn_J2d': './Model_inference/Mix_GCN/output/ctrgcn_V1_J/epoch1_test_score.pkl',
        'ctrgcn_B2d': './Model_inference/Mix_GCN/output/ctrgcn_V1_B/epoch1_test_score.pkl',
        'ctrgcn_JM2d': './Model_inference/Mix_GCN/output/ctrgcn_V1_JM/epoch1_test_score.pkl',
        'ctrgcn_BM2d': './Model_inference/Mix_GCN/output/ctrgcn_V1_BM/epoch1_test_score.pkl',
        'ctrgcn_J3d': './Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl',
        'ctrgcn_B3d': './Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl',
        'ctrgcn_JM3d': './Model_inference/Mix_GCN/output/ctrgcn_V1_JM_3D/epoch1_test_score.pkl',
        'ctrgcn_BM3d': './Model_inference/Mix_GCN/output/ctrgcn_V1_BM_3D/epoch1_test_score.pkl',
        'tdgcn_J2d': './Model_inference/Mix_GCN/output/tdgcn_V1_J/epoch1_test_score.pkl',
        'tdgcn_B2d': './Model_inference/Mix_GCN/output/tdgcn_V1_B/epoch1_test_score.pkl',
        'tdgcn_JM2d': './Model_inference/Mix_GCN/output/tdgcn_V1_JM/epoch1_test_score.pkl',
        'tdgcn_BM2d': './Model_inference/Mix_GCN/output/tdgcn_V1_BM/epoch1_test_score.pkl',
        'mstgcn_J2d': './Model_inference/Mix_GCN/output/mstgcn_V1_J/epoch1_test_score.pkl',
        'mstgcn_B2d': './Model_inference/Mix_GCN/output/mstgcn_V1_B/epoch1_test_score.pkl',
        'mstgcn_JM2d': './Model_inference/Mix_GCN/output/mstgcn_V1_JM/epoch1_test_score.pkl',
        'mstgcn_BM2d': './Model_inference/Mix_GCN/output/mstgcn_V1_BM/epoch1_test_score.pkl'
    }

    for key, path in score_files.items():
        parser.add_argument(f'--{key}_Score', type=str, default=path)

    parser.add_argument('--val_sample', type=str, default='./Process_data/CS_test_V1.txt')
    parser.add_argument('--benchmark', type=str, choices=['V1', 'V2'], default='V1')
    
    return parser

def calculate_score(files, weights, num_samples, num_classes):
    final_score = torch.zeros(num_samples, num_classes)
    for idx, file in enumerate(files):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        score = torch.tensor(pd.DataFrame(data).values.T)
        final_score += weights[idx] * score
    return final_score

def calculate_accuracy(pred_score, true_labels):
    _, pred_labels = torch.max(pred_score, dim=1)
    accuracy = (pred_labels == true_labels).float().mean().item()
    wrong_indices = (pred_labels != true_labels).nonzero(as_tuple=True)[0].tolist()
    print(f'Wrong predictions: {len(wrong_indices)} out of {len(true_labels)}')
    return accuracy

def generate_labels(file_path):
    labels = [int(name.split('A')[1][:3]) for name in np.loadtxt(file_path, dtype=str)]
    return torch.tensor(labels)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Score files
    files = [getattr(args, f"{key}_Score") for key in [
        'mixformer_J', 'mixformer_B', 'mixformer_JM', 'mixformer_BM', 'mixformer_k2', 'mixformer_k2M',
        'ctrgcn_J2d', 'ctrgcn_B2d', 'ctrgcn_JM2d', 'ctrgcn_BM2d', 'ctrgcn_J3d', 'ctrgcn_B3d', 
        'ctrgcn_JM3d', 'ctrgcn_BM3d', 'tdgcn_J2d', 'tdgcn_B2d', 'tdgcn_JM2d', 'tdgcn_BM2d', 
        'mstgcn_J2d', 'mstgcn_B2d', 'mstgcn_JM2d', 'mstgcn_BM2d'
    ]]

    # Set benchmark-specific parameters
    params = {
        'V1': {
            'num_classes': 155,
            'num_samples': 6307,
            'weights': [1.2, 0.77, 0.2, 0.2, 1.2, 1.2, 0.85, 1.2, 0.2, 0.2, 0.67, 0.87, 0.2, 0.2, 0.79, 1.2, 0.2, 0.2, 1.2, 1.2, 0.2, 0.2]
        },
        'V2': {
            'num_classes': 155,
            'num_samples': 6599,
            'weights': [0.72, 1.2, 0.2, 1.2, 1.2, 0.95, 1.2, 1.2, 0.2, 0.2, 1.2, 1.2, 0.2, 0.2, 1.2, 1.2, 0.2, 0.2, 0.67, 0.39, 0.2, 0.2]
        }
    }
    
    benchmark = params[args.benchmark]
    final_score = calculate_score(files, benchmark['weights'], benchmark['num_samples'], benchmark['num_classes'])
    true_labels = generate_labels(args.val_sample)
    accuracy = calculate_accuracy(final_score, true_labels)

    print(f'Accuracy: {accuracy:.4f}')
