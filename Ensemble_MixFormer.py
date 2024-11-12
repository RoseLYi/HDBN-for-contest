import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
from skopt import gp_minimize

def load_scores(file_path):
    """Load the score file if it exists, else return None."""
    if file_path:
        with open(file_path, 'rb') as file:
            return list(pickle.load(file).items())
    return None

def calculate_accuracy(weights, label, scores):
    """Calculate accuracy based on given weights and scores."""
    correct, total = 0, len(label)
    for i, lbl in enumerate(tqdm(label)):
        combined_score = sum(score[i][1] * w for score, w in zip(scores, weights))
        predicted = np.argmax(combined_score)
        correct += int(predicted == int(lbl))
    return correct / total

def objective(weights):
    """Objective function for optimizing weights."""
    accuracy = calculate_accuracy(weights, label, score_list)
    print(f"Accuracy: {accuracy:.4f}")
    return -accuracy  # Minimize negative accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA', 'csv1', 'csv2'}, help='Dataset choice')
    parser.add_argument('--alpha', type=float, default=1, help='Weighted summation factor')
    parser.add_argument('--joint-dir', help='Directory for joint evaluation results')
    parser.add_argument('--bone-dir', help='Directory for bone evaluation results')
    parser.add_argument('--joint-motion-dir', default=None)
    parser.add_argument('--bone-motion-dir', default=None)
    parser.add_argument('--joint-k2-dir', default=None)
    parser.add_argument('--joint-motion-k2-dir', default=None)
    args = parser.parse_args()

    # Load labels based on dataset
    dataset_paths = {
        'csv1': '/data/liujinfu/icmew/pose_data/V1.npz',
        'csv2': '/data/liujinfu/icmew/pose_data/V2.npz'
    }
    
    if args.dataset in dataset_paths:
        npz_data = np.load(dataset_paths[args.dataset])
        label = npz_data['y_test']
    else:
        raise NotImplementedError("Dataset not supported.")

    # Load score files
    score_files = [
        args.joint_dir,
        args.bone_dir,
        args.joint_motion_dir,
        args.bone_motion_dir,
        args.joint_k2_dir,
        args.joint_motion_k2_dir
    ]
    score_list = [load_scores(os.path.join(dir_path, 'epoch1_test_score.pkl')) for dir_path in score_files if dir_path]

    # If multiple score files are available, optimize weights
    if len(score_list) > 1:
        weight_space = [(0.2, 1.2)] * len(score_list)
        result = gp_minimize(objective, weight_space, n_calls=200, random_state=0)
        print(f'Maximum accuracy: {-result.fun * 100:.4f}%')
        print(f'Optimal weights: {result.x}')
