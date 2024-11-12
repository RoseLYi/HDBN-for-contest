
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import logging

# 配置日志记录
logging.basicConfig(filename='meta_learner_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    "degcn_B_3d": "../scores/Mix_GCN/degcn_B_3d.pkl"
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
    "skateformer_j_3d": "../scores/Mix_Former/skateformer_B_3d.pkl"
}

def calculate_class_weights(labels, num_classes=155):
    distro = np.bincount(labels, minlength=num_classes)
    weights = 1.0 - distro / len(labels)
    logging.info(f"Class distribution: {distro}")
    logging.info(f"Class weights: {weights}")
    return weights.astype(np.float32)

def load_model_data(paths, sample_size=2000):
    data_list = []
    for path in paths.values():
        with open(path, 'rb') as file:
            data_dict = pickle.load(file)
        data = np.array([data_dict[f"test_{i}"] for i in range(sample_size)])
        data_list.append(data)
    return np.array(data_list).transpose(1, 0, 2)

def load_features_and_labels(use_gcn=True, use_former=True):
    data_list = []
    if use_former:
        data_list.append(load_model_data(former_paths))
    if use_gcn:
        data_list.append(load_model_data(gcn_paths))
    X = np.sum(np.concatenate(data_list, axis=1), axis=1)
    y = np.load("test_label_A.npy")
    return X, y

def train_test_split_data(X, y, train_ratio=0.8):
    return train_test_split(X, y, train_size=train_ratio, random_state=42)

class MetaLearner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MetaLearner, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.float(), y_batch.long()
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)
    accuracy = correct / total
    return total_loss / len(loader), accuracy

def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.float(), y_batch.long()
            outputs = model(X_batch)
            total_loss += criterion(outputs, y_batch).item()
            correct += (outputs.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)
    accuracy = correct / total
    return total_loss / len(loader), accuracy

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma, self.weight, self.reduction = alpha, gamma, weight, reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-CE_loss)
        loss = self.alpha * ((1 - pt) ** self.gamma) * CE_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss

def main():
    X, y = load_features_and_labels()
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=32, shuffle=False)

    model = MetaLearner(input_dim=X.shape[1], output_dim=155)
    criterion = FocalLoss(weight=torch.from_numpy(calculate_class_weights(y_train)))
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    epochs = 50
    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer)
        eval_loss, eval_accuracy = evaluate(model, test_loader)
        logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_accuracy:.4f}")
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_accuracy:.4f}")
        scheduler.step(train_loss)

    torch.save(model.state_dict(), "meta_learner_weights.pth")
    logging.info("模型权重已保存为 meta_learner_weights.pth")
    print("模型权重已保存为 meta_learner_weights.pth")

if __name__ == "__main__":
    main()
