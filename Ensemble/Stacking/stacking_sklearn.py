
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

# 定义数据路径
gcn_names = {
    "ctrgcn_jm_3d": "../scores/Mix_GCN/ctrgcn_V1_JM_3d.pkl",
    "ctrgcn_b_3d": "../scores/Mix_GCN/ctrgcn_V1_B_3d.pkl",
    "ctrgcn_j_3d": "../scores/Mix_GCN/ctrgcn_V1_J_3d.pkl",
    "ctrgcn_j_3d_resample": "../scores/Mix_GCN/ctrgcn_V1_J_3d_resample.pkl",
    "ctrgcn_j_3d_resample_rotate": "../scores/Mix_GCN/ctrgcn_V1_J_3d_resample_rotate.pkl",
    "ctrgcn_b_2d": "../scores/Mix_GCN/ctrgcn_V1_B_2d.pkl",
    "ctrgcn_j_2d": "../scores/Mix_GCN/ctrgcn_V1_J_2d.pkl",
    "ctrgcn_bm_2d": "../scores/Mix_GCN/ctrgcn_V1_BM_2d.pkl",
    "ctrgcn_jm_2d": "../scores/Mix_GCN/ctrgcn_V1_JM_2d.pkl",
    "tdgcn_j_2d": "../scores/Mix_GCN/tdgcn_V1_J_2d.pkl",
    "blockgcn_j_3d": "../scores/Mix_GCN/blockgcn_J_3d.pkl",
    "blockgcn_jm_3d": "../scores/Mix_GCN/blockgcn_JM_3d.pkl",
    "blockgcn_b_3d": "../scores/Mix_GCN/blockgcn_B_3d.pkl",
    "blockgcn_bm_3d": "../scores/Mix_GCN/blockgcn_BM_3d.pkl",
    "ctrgcn_b_3d_resample_rotate": "../scores/Mix_GCN/ctrgcn_V1_B_3d_resample_rotate.pkl",
    "degcn_J_3d": "../scores/Mix_GCN/degcn_J_3d.pkl",
    "degcn_B_3d": "../scores/Mix_GCN/degcn_B_3d.pkl",
    "tegcn_V1_J_3d": "../scores/Mix_GCN/tegcn_V1_J_3d.pkl"
}

former_names = {
    "former_bm_r_w_2d": "../scores/Mix_Former/mixformer_BM_r_w_2d.pkl",
    "former_bm_2d": "../scores/Mix_Former/mixformer_BM_2d.pkl",
    "former_j_2d": "../scores/Mix_Former/mixformer_J_2d.pkl",
    "former_j_3d": "../scores/Mix_Former/mixformer_J_3d.pkl",
    "former_b_3d": "../scores/Mix_Former/mixformer_B_3d.pkl",
    "former_j_3d_resample_rotate": "../scores/Mix_Former/mixformer_J_3d_resample_rotate.pkl",
    "former_jm_2d": "../scores/Mix_Former/mixformer_JM_2d.pkl",
    "former_b_3d_resample_rotate": "../scores/Mix_Former/mixformer_B_3d_resample_rotate.pkl",
    "skateformer_j_3d": "../scores/Mix_Former/skateformer_B_3d.pkl",
}

def load_model_scores(file_paths, sample_size=2000):
    data_list = []
    for path in file_paths.values():
        with open(path, 'rb') as file:
            data_dict = pickle.load(file)
        data = np.array([data_dict.get(f"test_{i}") for i in range(sample_size)])
        data_list.append(data)
    return np.array(data_list)

def load_data(use_gcn=True, use_former=True, label_path="test_label_A.npy"):
    feature_sets = []
    if use_gcn:
        feature_sets.append(load_model_scores(gcn_paths))
    if use_former:
        feature_sets.append(load_model_scores(former_paths))
        
    X = np.concatenate(feature_sets, axis=1).transpose(1, 0, 2).reshape(-1, len(feature_sets) * 155)
    y = np.load(label_path)
    return X, y

def build_stacking_model(base_estimators, final_estimator=LogisticRegression()):
    return StackingClassifier(estimators=base_estimators, final_estimator=final_estimator)

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Stacking Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy

if __name__ == "__main__":
    X, y = load_data(use_gcn=True, use_former=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    base_learners = [('lr', LogisticRegression(max_iter=1000))]
    stacking_model = build_stacking_model(base_learners)
    stacking_model.fit(X_train, y_train)
    evaluate_model(stacking_model, X_test, y_test)
