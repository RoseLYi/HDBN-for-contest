import argparse
import pickle
import numpy as np
from tqdm import tqdm
from skopt import gp_minimize

def load_score(file_path):
    """Load the score file if the path is provided, otherwise return None."""
    with open(file_path, 'rb') as file:
        return list(pickle.load(file).items())

def objective(weights):
    """Objective function to maximize accuracy using weighted score fusion."""
    right_num = 0
    total_num = len(label)

    for i in tqdm(range(total_num)):
        l = label[i]
        combined_score = sum(r[i][1] * w for r, w in zip(scores, weights))
        pred_label = np.argmax(combined_score)
        right_num += int(pred_label == int(l))

    accuracy = right_num / total_num
    print(f"Accuracy: {accuracy:.4f}")
    return -accuracy  # Minimize negative accuracy for maximizing accuracy

def get_parser():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Multi-stream ensemble with weight optimization")
    score_files = [
        'mixformer_J', 'mixformer_B', 'mixformer_JM', 'mixformer_BM', 'mixformer_k2', 'mixformer_k2M',
        'ctrgcn_J2d', 'ctrgcn_B2d', 'ctrgcn_JM2d', 'ctrgcn_BM2d', 'ctrgcn_J3d', 'ctrgcn_B3d', 
        'ctrgcn_JM3d', 'ctrgcn_BM3d', 'tdgcn_J2d', 'tdgcn_B2d', 'tdgcn_JM2d', 'tdgcn_BM2d', 
        'mstgcn_J2d', 'mstgcn_B2d', 'mstgcn_JM2d', 'mstgcn_BM2d'
    ]
    
    for score_file in score_files:
        parser.add_argument(f'--{score_file}_Score', type=str, required=True, help=f'Path to {score_file} score file')
    
    parser.add_argument('--benchmark', choices=['V1', 'V2'], default='V1', help='Benchmark dataset version')
    return parser

if __name__ == "__main__":
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # Load labels
    dataset_path = f'./Model_inference/Mix_Former/dataset/save_2d_pose/{args.benchmark}.npz'
    npz_data = np.load(dataset_path)
    label = npz_data['y_test']

    # Load score files
    score_files = [getattr(args, f"{name}_Score") for name in get_parser().parse_args()._get_kwargs() if 'Score' in name]
    scores = [load_score(file) for file in score_files]

    # Optimize weights
    space = [(0.2, 1.2)] * len(scores)
    result = gp_minimize(objective, space, n_calls=200, random_state=0)
    print(f'Maximum accuracy: {-result.fun * 100:.4f}%')
    print(f'Optimal weights: {result.x}')

