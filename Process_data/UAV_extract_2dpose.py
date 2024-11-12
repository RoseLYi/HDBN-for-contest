import os
import argparse
import numpy as np

# 定义训练集ID
CS_train_V1 = [0, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
               21, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
               42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 59, 
               61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 73, 76, 77, 78, 79, 80, 
               81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 98, 100, 102, 103, 105, 
               106, 110, 111, 112, 114, 115, 116, 117, 118]

CS_train_V2 = [0, 3, 4, 5, 6, 8, 10, 11, 12, 14, 16, 18, 19, 20, 21, 22, 24, 
               26, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 
               49, 52, 54, 56, 57, 59, 60, 61, 62, 63, 64, 66, 67, 69, 70, 71, 
               72, 73, 74, 75, 77, 78, 79, 80, 81, 83, 84, 86, 87, 88, 89, 91, 
               92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 106, 107, 
               108, 109, 110, 111, 112, 113, 114, 115, 117, 118]

def extract_pose(file_path: str) -> np.ndarray:
    with open(file_path, 'r') as f:
        num_frames = int(f.readline())
        joint_data = []
        for _ in range(num_frames):
            num_bodies = int(f.readline())
            frame_data = np.zeros((num_bodies, 17, 2))
            for body in range(num_bodies):
                f.readline()  # 跳过行
                num_joints = int(f.readline())
                assert num_joints == 17
                for joint in range(num_joints):
                    xy = np.array(f.readline().split()[:2], dtype=np.float64)
                    frame_data[body, joint] = xy
            joint_data.append(frame_data)
    return np.array(joint_data)

def get_max_frame_count(root_path: str, sample_files: list) -> int:
    max_frame = 0
    for file in sample_files:
        with open(os.path.join(root_path, file), 'r') as f:
            frame_count = int(f.readline())
            max_frame = max(max_frame, frame_count)
    return max_frame

def project_to_2d(joint_data):
    return joint_data[:, :2, :, :, :]

def main(data_path: str, label_path: str) -> None:
    joint_data = np.load(data_path)
    labels = np.load(label_path)
    joint_data_2d = project_to_2d(joint_data)
    np.save('./save_2d_pose/test_joint_B_2d.npy', joint_data_2d)
    np.save('./save_2d_pose/test_labels_B.npy', labels)
    print("转换为2D数据并保存成功")

def get_parser():
    parser = argparse.ArgumentParser(description='Extract 2D pose data from dataset')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--label_path', type=str, required=True)
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.data_path, args.label_path)
