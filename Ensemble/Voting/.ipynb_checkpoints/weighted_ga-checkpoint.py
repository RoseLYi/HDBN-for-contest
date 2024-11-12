
import pickle
import numpy as np
from deap import base, creator, tools, algorithms
import time
import random
import multiprocessing
from functools import partial

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
    "tegcn_V1_J_3d": "../scores/Mix_GCN/tegcn_V1_J_3d.pkl",
    "tegcn_V1_B_3d": "../scores/Mix_GCN/tegcn_V1_B_3d.pkl"
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

# 加载模型数据
def load_data(gcn=False, former=False):
    data = []
    if gcn:
        data += load_from_paths(model_paths["gcn"])
    if former:
        data += load_from_paths(model_paths["former"])
    X = np.array(data).transpose(1, 0, 2)
    y = np.load("test_label_A.npy")
    return X, y

def load_from_paths(paths):
    return [load_model(path) for path in paths]

def load_model(path):
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)
    return [data_dict[f"test_{i}"] for i in range(2000)]

def softmax(X):
    return np.exp(X) / np.sum(np.exp(X), axis=-1, keepdims=True)

def weighted_predictions(X, weights):
    return np.array([np.argmax(sum(weights[i] * softmax(X[sample][i]) for i in range(len(weights)))) 
                     for sample in range(len(X))])

def loss_function(weights, X, y):
    predictions = weighted_predictions(X, weights)
    return -np.mean(predictions == y)

def evaluate_fitness(individual, X, y):
    return -loss_function(individual, X, y),

def optimize_weights_ga(X, y, generations=30, pop_size=60):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=X.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", partial(evaluate_fitness, X=X, y=y))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.6, sigma=0.6, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    population = toolbox.population(n=pop_size)
    hall_of_fame = tools.HallOfFame(1)

    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, 
                        halloffame=hall_of_fame, verbose=True)

    pool.close()
    pool.join()
    return hall_of_fame[0]

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

if __name__ == "__main__":
    X, y = load_data(gcn=True, former=True)
    start_time = time.time()
    
    # 优化权重
    optimized_weights = optimize_weights_ga(X, y, generations=30, pop_size=60)
    print(f"Optimization completed in {time.time() - start_time:.2f} seconds")
    print(f"Optimized Weights (GA): {optimized_weights}")

    # 使用优化权重进行预测
    predictions = weighted_predictions(X, optimized_weights)
    acc = accuracy(y, predictions)
    print(f"Accuracy with Optimized Weights (GA): {acc}%")
