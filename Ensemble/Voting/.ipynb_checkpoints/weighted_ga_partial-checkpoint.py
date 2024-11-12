
import pickle
import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from deap import base, creator, tools, algorithms
import time
import random
import multiprocessing
from functools import partial

# 定义模型文件路径

gcn_3d_names = {
    "ctrgcn_jm_3d": "../scores/Mix_GCN/ctrgcn_V1_JM_3d.pkl",
    "ctrgcn_b_3d": "../scores/Mix_GCN/ctrgcn_V1_B_3d.pkl",
    "ctrgcn_j_3d": "../scores/Mix_GCN/ctrgcn_V1_J_3d.pkl",
    "ctrgcn_j_3d_resample": "../scores/Mix_GCN/ctrgcn_V1_J_3d_resample.pkl",
    "ctrgcn_j_3d_resample_rotate": "../scores/Mix_GCN/ctrgcn_V1_J_3d_resample_rotate.pkl",
    "blockgcn_j_3d": "../scores/Mix_GCN/blockgcn_J_3d.pkl",
    "blockgcn_jm_3d": "../scores/Mix_GCN/blockgcn_JM_3d.pkl",
    "blockgcn_b_3d": "../scores/Mix_GCN/blockgcn_B_3d.pkl",
    "blockgcn_bm_3d": "../scores/Mix_GCN/blockgcn_BM_3d.pkl",
    "ctrgcn_b_3d_resample_rotate": "../scores/Mix_GCN/ctrgcn_V1_B_3d_resample_rotate.pkl",
    "degcn_J_3d": "../scores/Mix_GCN/degcn_J_3d.pkl"
}

gcn_2d_names = {
    "ctrgcn_b_2d": "../scores/Mix_GCN/ctrgcn_V1_B_2d.pkl",
    "ctrgcn_j_2d": "../scores/Mix_GCN/ctrgcn_V1_J_2d.pkl",
    "ctrgcn_bm_2d": "../scores/Mix_GCN/ctrgcn_V1_BM_2d.pkl",
    "ctrgcn_jm_2d": "../scores/Mix_GCN/ctrgcn_V1_JM_2d.pkl",
    "tdgcn_j_2d": "../scores/Mix_GCN/tdgcn_V1_J_2d.pkl",
}

former_3d_names = {
    "former_j_3d": "../scores/Mix_Former/mixformer_J_3d.pkl",
    "former_b_3d": "../scores/Mix_Former/mixformer_B_3d.pkl",
    "former_j_3d_resample_rotate": "../scores/Mix_Former/mixformer_J_3d_resample_rotate.pkl",
}

former_2d_names = {
    "former_bm_r_w_2d": "../scores/Mix_Former/mixformer_BM_r_w_2d.pkl",
    "former_bm_2d": "../scores/Mix_Former/mixformer_BM_2d.pkl",
    "former_j_2d": "../scores/Mix_Former/mixformer_J_2d.pkl",
    "former_jm_2d": "../scores/Mix_Former/mixformer_JM_2d.pkl",
}

def load_model_data(paths):
    """从给定路径加载模型数据，返回数组形式的数据。"""
    data = []
    for path in paths:
        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
            model_data = [data_dict[f"test_{i}"] for i in range(2000)]
            data.append(model_data)
    return np.array(data)

def load_data(gcn_2d=False, gcn_3d=False, former_2d=False, former_3d=False):
    """加载指定的模型数据，并返回特征矩阵和标签。"""
    data_sources = []
    if gcn_2d:
        data_sources.append(load_model_data(model_files["gcn_2d"]))
    if gcn_3d:
        data_sources.append(load_model_data(model_files["gcn_3d"]))
    if former_2d:
        data_sources.append(load_model_data(model_files["former_2d"]))
    if former_3d:
        data_sources.append(load_model_data(model_files["former_3d"]))

    X = np.concatenate(data_sources, axis=0).transpose(1, 0, 2)
    y = np.load("test_label_A.npy")
    return X, y

def softmax(X):
    """对特征矩阵 X 的每个向量应用 softmax。"""
    return np.exp(X) / np.sum(np.exp(X), axis=-1, keepdims=True)

def weighted_predictions(X, weights):
    """根据权重对模型预测进行加权并返回最终预测。"""
    predictions = []
    for sample in X:
        weighted_sum = np.sum([weights[i] * softmax(sample[i]) for i in range(len(weights))], axis=0)
        predictions.append(np.argmax(weighted_sum))
    return np.array(predictions)

def loss_function(weights, X, y):
    """计算权重的损失函数，即负准确率。"""
    predictions = weighted_predictions(X, weights)
    return -np.mean(predictions == y)

def optimize_weights_ga(X, y, generations=25, pop_size=50):
    """使用遗传算法优化加权参数。"""
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=X.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", partial(evaluate, X=X, y=y))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    population = toolbox.population(n=pop_size)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=True)

    best_individual = tools.selBest(population, k=1)[0]
    pool.close()
    pool.join()
    return best_individual

def evaluate(individual, X, y):
    """评估个体在数据上的表现，返回准确率的负值。"""
    return -loss_function(individual, X, y),

def save_weighted_results(data, weights, model_type="gcn_2d"):
    """保存加权后的预测结果至指定文件。"""
    weighted_data = np.sum([weights[i] * data[i] for i in range(len(weights))], axis=0)
    results = {f"test_{i}": weighted_data[i] for i in range(weighted_data.shape[0])}

    with open(f"./partial/partial_{model_type}.pkl", "wb") as f:
        pickle.dump(results, f)

def accuracy_score(y_true, y_pred):
    """计算并返回准确率。"""
    return np.mean(y_true == y_pred) * 100

if __name__ == "__main__":
    gcn_2d, gcn_3d, former_2d, former_3d = False, False, False, False

    # 加载数据
    X, y = load_data(gcn_2d=gcn_2d, gcn_3d=gcn_3d, former_2d=former_2d, former_3d=former_3d)
    
    # 使用遗传算法优化权重
    start_time = time.time()
    optimized_weights = optimize_weights_ga(X, y)
    print(f"Optimization completed in {time.time() - start_time:.2f} seconds")
    print(f"Optimized Weights: {optimized_weights}")

    # 使用优化的权重进行加权预测
    predictions = weighted_predictions(X, optimized_weights)
    acc = accuracy_score(y, predictions)
    print(f"Accuracy with Optimized Weights: {acc}%")

    # 保存加权结果
    save_weighted_results(X, optimized_weights, model_type="gcn_2d" if gcn_2d else "former_2d")
