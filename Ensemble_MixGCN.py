import argparse
import torch
import pickle
import numpy as np
import pandas as pd

def get_parser():
    """Defines and returns the argument parser."""
    parser = argparse.ArgumentParser(description="Multi-stream ensemble")
    
    score_files = {
        'ctrgcn_J2d_Score': './Model_inference/Mix_GCN/output/ctrgcn_V1_J/epoch1_test_score.pkl',
        'ctrgcn_B2d_Score': './Model_inference/Mix_GCN/output/ctrgcn_V1_B/epoch1_test_score.pkl',
        'ctrgcn_JM2d_Score': './Model_inference/Mix_GCN/output/ctrgcn_V1_JM/epoch1_test_score.pkl',
        'ctrgcn_BM2d_Score': './Model_inference/Mix_GCN/output/ctrgcn_V1_BM/epoch1_test_score.pkl',
        'ctrgcn_J3d_Score': './Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl',
        'ctrgcn_B3d_Score': './Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl',
        'ctrgcn_JM3d_Score': './Model_inference/Mix_GCN/output/ctrgcn_V1_JM_3D/epoch1_test_score.pkl',
        'ctrgcn_BM3d_Score': './Model_inference/Mix_GCN/output/ctrgcn_V1_BM_3D/epoch1_test_score.pkl',
        'tdgcn_J2d_Score': './Model_inference/Mix_GCN/output/tdgcn_V1_J/epoch1_test_score.pkl',
        'tdgcn_B2d_Score': './Model_inference/Mix_GCN/output/tdgcn_V1_B/epoch1_test_score.pkl',
        'tdgcn_JM2d_Score': './Model_inference/Mix_GCN/output/tdgcn_V1_JM/epoch1_test_score.pkl',
        'tdgcn_BM2d_Score': './Model_inference/Mix_GCN/output/tdgcn_V1_BM/epoch1_test_score.pkl',
        'mstgcn_J2d_Score': './Model_inference/Mix_GCN/output/mstgcn_V1_J/epoch1_test_score.pkl',
        'mstgcn_B2d_Score': './Model_inference/Mix_GCN/output/mstgcn_V1_B/epoch1_test_score.pkl',
        'mstgcn_JM2d_Score': './Model_inference/Mix_GCN/output/mstgcn_V1_JM/epoch1_test_score.pkl',
        'mstgcn_BM2d_Score': './Model_inference/Mix_GCN/output/mstgcn_V1_BM/epoch1_test_score.pkl'
    }
    
    for name, path in score_files.items():
        parser.add_argument(f'--{name}', type=str, default=path, help=f'Path to {name}')
    
    parser.add_argument('--val_sample', type=str, default='./Process_data/CS_test_V1.txt')
    parser.add_argument('--benchmark', type=str, choices=['V1', 'V2'], default='V1')
    
    return parser

def load_scores(file_paths, rates, sample_num, num_classes):
    """Calculates weighted scores from multiple files."""
    final_score = torch.zeros(sample_num, num_classes)
    for idx, file_path in enumerate(file_paths):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        score = torch.tensor(pd.DataFrame(data).T.values)
        final_score += rates[idx] * score
    return final_score

def calculate_accuracy(final_score, true_labels):
    """Calculates accuracy and returns it."""
    _, pred_labels = torch.max(final_score, 1)
    accuracy = (pred_labels == true_labels).float().mean().item()
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy

def load_labels(file_path):
    """Loads labels from a file and returns them as a tensor."""
    labels = [int(name.split('A')[1][:3]) for name in np.loadtxt(file_path, dtype=str)]
    return torch.tensor(labels)

if __name__ == "__main__":
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # Collect file paths from arguments
    file_paths = [getattr(args, f'{name}_Score') for name in [
        'ctrgcn_J2d', 'ctrgcn_B2d', 'ctrgcn_JM2d', 'ctrgcn_BM2d',
        'ctrgcn_J3d', 'ctrgcn_B3d', 'ctrgcn_JM3d', 'ctrgcn_BM3d',
        'tdgcn_J2d', 'tdgcn_B2d', 'tdgcn_JM2d', 'tdgcn_BM2d',
        'mstgcn_J2d', 'mstgcn_B2d', 'mstgcn_JM2d', 'mstgcn_BM2d'
    ]]

    # Benchmark-specific parameters
    params = {
        'V1': {
            'num_classes': 155,
            'sample_num': 6307,
            'rates': [0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3]
        },
        'V2': {
            'num_classes': 155,
            'sample_num': 6599,
            'rates': [0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.05, 0.05, 0.05, 0.05]
        }
    }

    benchmark = params[args.benchmark]
    final_score = load_scores(file_paths, benchmark['rates'], benchmark['sample_num'], benchmark['num_classes'])
    true_labels = load_labels(args.val_sample)
    calculate_accuracy(final_score, true_labels)
