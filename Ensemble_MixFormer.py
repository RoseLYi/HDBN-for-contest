import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm
from skopt import gp_minimize

def objective(weights):
    right_num = total_num = 0
    for i in tqdm(range(len(label))):
        l = label[i]
        _, r11 = r1[i]
        _, r22 = r2[i]
        _, r33 = r3[i]
        _, r44 = r4[i]
        _, r55 = r5[i]
        _, r66 = r6[i]
        _, r77 = r7[i]
        _, r88 = r8[i]
        _, r99 = r9[i]
        _, r1010 = r10[i]
        
        r = r11 * weights[0] + r22 * weights[1] + r33 * weights[2] + r44 * weights[3] + r55 * weights[4] + r66 * weights[5]+ r77 * weights[6] + r88 * weights[7] + r99 * weights[8] + r1010 * weights[9]
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    print(acc)
    return -acc  # We want to maximize accuracy, hence minimize -accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'csv1','csv2'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir', default=None)
    parser.add_argument('--bone-motion-dir', default=None)
    parser.add_argument('--joint-k2-dir', default=None)
    parser.add_argument('--joint-motion-k2-dir', default=None)
    parser.add_argument('--joint-dir2',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir2',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir2', default=None)
    parser.add_argument('--bone-motion-dir2', default=None)


    arg = parser.parse_args()

    dataset = arg.dataset
    if 'csv1' in arg.dataset:
        npz_data = np.load('/data/liujinfu/icmew/pose_data/V1.npz')
        label = npz_data['y_test']#np.where(npz_data['y_test'] > 0)[1]
    elif 'csv2' in arg.dataset:
        npz_data = np.load('/data/liujinfu/icmew/pose_data/V2.npz')
        label = npz_data['y_test']#np.where(npz_data['y_test'] > 0)[1]

    else:
        raise NotImplementedError

    with open(os.path.join(arg.joint_dir, 'epoch1_test_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(arg.bone_dir, 'epoch1_test_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    if arg.joint_motion_dir is not None:
        with open(os.path.join(arg.joint_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r3:
            r3 = list(pickle.load(r3).items())
    if arg.bone_motion_dir is not None:
        with open(os.path.join(arg.bone_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r4:
            r4 = list(pickle.load(r4).items())

    if arg.joint_k2_dir is not None:
        with open(os.path.join(arg.joint_k2_dir, 'epoch1_test_score.pkl'), 'rb') as r5:
            r5 = list(pickle.load(r5).items())
    if arg.joint_motion_k2_dir is not None:
        with open(os.path.join(arg.joint_motion_k2_dir, 'epoch1_test_score.pkl'), 'rb') as r6:
            r6 = list(pickle.load(r6).items())

    with open(os.path.join(arg.joint_dir2, 'epoch1_test_score.pkl'), 'rb') as r7:
        r7 = list(pickle.load(r7).items())

    with open(os.path.join(arg.bone_dir2, 'epoch1_test_score.pkl'), 'rb') as r8:
        r8 = list(pickle.load(r8).items())

    if arg.joint_motion_dir is not None:
        with open(os.path.join(arg.joint_motion_dir2, 'epoch1_test_score.pkl'), 'rb') as r9:
            r9 = list(pickle.load(r9).items())
    if arg.bone_motion_dir is not None:
        with open(os.path.join(arg.bone_motion_dir2, 'epoch1_test_score.pkl'), 'rb') as r10:
            r10 = list(pickle.load(r10).items())
            
    if arg.joint_motion_dir is not None and arg.bone_motion_dir is not None:
        space = [(0.2, 1.2) for i in range(10)]
        result = gp_minimize(objective, space, n_calls=200, random_state=0)
        print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
        print('Optimal weights: {}'.format(result.x))



