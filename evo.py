import numpy as np
import random
import json
import scipy.stats as ss
import multiprocessing as mp
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_theme()
# sns.color_palette()

BIT_LENGTH = 16
HALF_LENGTH = 8
LAYERS = 4
MUTATION_RATE = 0.05
POP_SIZE = 50
GENERATIONS = 2000
DOMAIN = 0

def fitness_calculate(bit_slice):
    if DOMAIN == 0: # Convex Bits
        total_fitness = np.sum(bit_slice, axis=1) * 2
        return total_fitness
    elif DOMAIN == 1: # Hashed Bits
        random_vals = np.clip(np.abs(np.random.normal(loc=0, scale=(4/3), size=(bit_slice.shape[0],))), 0, 5)
        total_fitness = np.power(2, random_vals)
        return total_fitness
    elif DOMAIN == 2: # Deceptive Bits
        total_fitness = np.zeros(bit_slice.shape[0])
        for n in range(LAYERS):
            layer_iter = pow(2, n)
            layer_len = pow(2, LAYERS-n)
            for j in range(0, layer_len, 2):
                equals = np.not_equal(bit_slice[:, (j+0)*layer_iter:(j+1)*layer_iter], bit_slice[:, (j+1)*layer_iter:(j+2)*layer_iter])
                total_match = np.sum(equals, axis=1)
                total_fitness += np.equal(total_match, 0) * layer_iter
        return total_fitness

def mutate(bit_slice):
    bits = np.copy(bit_slice)
    rand_genome = np.random.rand(POP_SIZE, BIT_LENGTH) < 0.5
    adj_genome = np.random.rand(POP_SIZE, BIT_LENGTH) < MUTATION_RATE
    bits[adj_genome] = rand_genome[adj_genome]
    return bits

def fitness_calculate_fast(bit_slice, fitness_array):
    ints = BitsToIntAFast(bit_slice)
    return fitness_array[ints]
def BitsToIntAFast(bits):
    m,n = bits.shape # number of columns is needed, not bits.size
    a = 2**np.arange(n)[::-1]  # -1 reverses array of powers of 2 of same length as bits
    return bits @ a  # this matmult is the key line of code

def hammings_calc(population):
    return (population[:, None, :] != population).sum(2)

def hammings_metric(hammings, population_metrics):
    smallests = np.zeros((POP_SIZE*2, 5), dtype=np.uint8)
    for n in range(POP_SIZE * 2):
        smallests[n] = np.argpartition(hammings[n], 5)[:5]
    distances = np.take_along_axis(hammings, smallests, axis=1)
    population_metrics[:, 3] = np.sum(distances, axis=1)

def compute_metrics(population_metrics, population, i):
    ranked = ss.rankdata(population_metrics[:, 0])
    norm = (ranked-1) / (len(ranked)-1)
    population_metrics[:, 1] = norm
    hammings = hammings_calc(population)
    hammings_metric(hammings, population_metrics)

    population_metrics[:, 4] = np.random.random(POP_SIZE * 2)
    population_metrics[:, 5] = i / GENERATIONS

# Given a selection function, and a fitness landscape, evolve a population.
def run(sel_func, fitness_array):
    log = []

    random_bit_flop = np.random.choice([True, False], size=(BIT_LENGTH))

    # Instantiate random population.
    # population = np.random.choice([True, False], size=(POP_SIZE*2, BIT_LENGTH))
    population = np.stack([np.random.choice([True, False], size=(BIT_LENGTH))] * POP_SIZE*2, axis=0)
    population_metrics = np.zeros((POP_SIZE*2, 6), dtype=float)
    population_metrics[:POP_SIZE, 0] = fitness_calculate_fast(population[:POP_SIZE], fitness_array)

    for i in range(GENERATIONS):

        # Increase age
        population_metrics[:POP_SIZE, 2] += 1

        # Make children
        population[POP_SIZE:] = mutate(population[:POP_SIZE])
        flopped_population = np.logical_xor(population[POP_SIZE:], random_bit_flop)
        population_metrics[POP_SIZE:, 0] = fitness_calculate_fast(flopped_population, fitness_array)

        # Compute metrics for every individual.
        compute_metrics(population_metrics, population, i)

        # Compute internal fitness with selection function
        sel = sel_func(population_metrics, population)

        for n in range(POP_SIZE):
            child_n = POP_SIZE + n;
            comp_id = random.randrange(POP_SIZE)
            if sel[child_n] >= sel[comp_id]:
                population[comp_id] = population[child_n]
                population_metrics[comp_id] = population_metrics[child_n]

        population[POP_SIZE:] = 0
        population_metrics[POP_SIZE:] = 0
        log.append(np.mean(population_metrics[:POP_SIZE], axis=0).tolist())

    return log

class GroupGenome(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
        self.layer1 = nn.Linear(6, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, 1)

    def forward(self, ob):
        ob[:, 0] /= 32
        ob[:, 2] /= 50
        with torch.no_grad():
            layer1 = self.tanh(self.layer1(ob))
            layer2 = self.tanh(self.layer2(layer1))
            layer3 = self.tanh(self.layer3(layer2))
            return layer3

#     def __init__(self):
#         super().__init__()
#         self.tanh = nn.Tanh()
#         self.layer1 = nn.Linear(6, 1)

#     def forward(self, ob):
#         ob[:, 0] /= 32
#         ob[:, 2] /= 50
#         with torch.no_grad():
#             layer1 = self.tanh(self.layer1(ob))
#             return layer1

    def sample_noise(self):
        nn_noise = []
        for n in self.parameters():
            noise = np.random.normal(size=n.data.numpy().shape)
            nn_noise.append(noise)
        return nn_noise

    def get_parameters(self):
        params = []
        for p in self.parameters():
            params.append(p.data.numpy())
        return params

    def set_parameters(self, params):
        for p_in, p in zip(params, self.parameters()):
            p.data = torch.FloatTensor(p_in)
            

class GroupGenomeLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
        self.layer1 = nn.Linear(6, 1)

    def forward(self, ob):
        ob[:, 0] /= 32
        ob[:, 2] /= 50
        with torch.no_grad():
            layer1 = self.tanh(self.layer1(ob))
            return layer1

    def sample_noise(self):
        nn_noise = []
        for n in self.parameters():
            noise = np.random.normal(size=n.data.numpy().shape)
            nn_noise.append(noise)
        return nn_noise

    def get_parameters(self):
        params = []
        for p in self.parameters():
            params.append(p.data.numpy())
        return params

    def set_parameters(self, params):
        for p_in, p in zip(params, self.parameters()):
            p.data = torch.FloatTensor(p_in)


def worker(genome_queue, result_queue, worker_id):
    np.random.seed(worker_id)
    genome_net = GroupGenome()
    print("Worker %d starting" % worker_id)

    # Making fitness array
    fitness_array = np.zeros((pow(2, 16)), dtype=np.uint8)
    lower_bits = np.unpackbits(np.arange(256, dtype=np.uint8)).reshape((256, 8))
    for i in range(256):
        higher_bits = np.unpackbits(np.ones(256, dtype=np.uint8) * i).reshape((256, 8))
        full_bits = np.concatenate([higher_bits, lower_bits], axis=1)
        fitness = fitness_calculate(full_bits)
        fitness_array[i * 256: (i+1) * 256] = fitness

    while True:
        params = genome_queue.get()
        genome_net.set_parameters(params)
        genome_sel = lambda metrics : genome_net.forward(torch.tensor(metrics).float()).numpy()
        run_log = run(genome_sel, fitness_array)
        gfit = run_log[GENERATIONS-1][0]
        result_queue.put((params, gfit))

def evolve_group(name):
    genome_net = GroupGenome()

    genome_queue = mp.Queue()
    result_queue = mp.Queue()
    processes = []
    for i in range(10):
        p = mp.Process(target=worker, args=(genome_queue, result_queue, i))
        p.start()
        processes.append(p)

    for i in range(1000):
        # Run trials on mutated selection functions.
        current_params = genome_net.get_parameters()
        noises = []
        group_fitness = []

        for j in range(20):
            new_params = []
            noise = genome_net.sample_noise()
            for p in range(len(current_params)):
                new_params.append(current_params[p] + noise[p] * 0.05)
            genome_queue.put(new_params)
        for j in range(20):
            noise, gfit = result_queue.get()
            noises.append(noise)
            group_fitness.append(gfit)

        print(group_fitness)
        print(i, np.mean(group_fitness))
        # Update params based on weighted average.
        ranked = ss.rankdata(group_fitness)
        norm = (ranked-1) / (len(ranked)-1)
        norm -= 0.5
        new_params = []
        for p in range(len(current_params)):
            new_params.append(current_params[p])
            for j in range(len(norm)):
                new_params[p] += noises[j][p] * norm[j] * 0.05
        genome_net.set_parameters(current_params)
        if i+1 == 10:
            torch.save(genome_net.state_dict(), "res/" + name + "10.pt")
        if i+1 == 1000:
            torch.save(genome_net.state_dict(), "res/" + name + "1k.pt")
    for p in processes:
        p.terminate()
        p.join()
        
def map_elite(is_cma=False):
    from qdpy import algorithms, containers, plots, tools
    from qdpy.base import ParallelismManager
    import math
    from functools import partial
    
    fitness_array = np.zeros((pow(2, 16)), dtype=np.uint8)
    lower_bits = np.unpackbits(np.arange(256, dtype=np.uint8)).reshape((256, 8))
    for i in range(256):
        higher_bits = np.unpackbits(np.ones(256, dtype=np.uint8) * i).reshape((256, 8))
        full_bits = np.concatenate([higher_bits, lower_bits], axis=1)
        fitness = fitness_calculate(full_bits)
        fitness_array[i * 256: (i+1) * 256] = fitness
        
    random_bit_flop = np.random.choice([True, False], size=(BIT_LENGTH))

    def eval_fn(ind):
        # print("Eval {}".format(ind))
        np_ind = np.array(ind)
        np_fit = np.bitwise_xor(np_ind, random_bit_flop)
        """An example evaluation function. It takes an individual as input, and returns the pair ``(fitness, features)``, where ``fitness`` and ``features`` are sequences of scores."""
        fits = fitness_calculate_fast(np_fit[None], fitness_array)
        metric1 = np.sum(np_ind) / BIT_LENGTH
        metric2 = 0.5 + ((np.sum(np_ind[:HALF_LENGTH]) - np.sum(np_ind[HALF_LENGTH:])) / BIT_LENGTH)
        to_ret = (fits[0],), (metric1, metric2)
        # print(to_ret)
        return to_ret
    
    def vary(ind):
        # print("Vary {}".format(ind))
        for i in range(len(ind)):
            if random.random() < MUTATION_RATE:
                ind[i] = np.random.choice([0, 1])
        return ind
    
    starter_genome = [np.random.choice([0,1]) for _ in range(BIT_LENGTH)]

    init_fn = lambda x: starter_genome.copy()
    select_or_initialise = partial(tools.sel_or_init,
            sel_fn = tools.sel_random,
            sel_pb = 1.0,
            init_fn = init_fn,
            init_pb = 0.0)

    # Create container and algorithm. Here we use MAP-Elites, by illuminating a Grid container by evolution.
    grid = containers.Grid(shape=(7,7), max_items_per_bin=1, fitness_domain=((0, 32),), features_domain=((0., 1.), (0., 1.)))
    
    if is_cma:
        algo = algorithms.CMAES(grid, budget=POP_SIZE*2000, batch_size=POP_SIZE, optimisation_task="maximisation", 
        select_or_initialise=select_or_initialise, vary=vary)
    else:
        algo = algorithms.Evolution(grid, budget=POP_SIZE*2000, batch_size=POP_SIZE, optimisation_task="maximisation", 
        select_or_initialise=select_or_initialise, vary=vary)
    
    fit_histories = []
    
    def cb(a, b):
        fit_histories.append(algo.container.best_fitness[0])
        
    algo.add_callback("iteration", cb)

    # Create a logger to pretty-print everything and generate output data files
    # logger = algorithms.TQDMAlgorithmLogger(algo, log_base_path='mapres/')

    # Run illumination process !
    with ParallelismManager("none") as pMgr:
        best = algo.optimise(eval_fn, executor = pMgr.executor, batch_mode=False)

    # Print results info
    # print("\n" + algo.summary())

    # Plot the results
    # plots.default_plots_grid(logger)
    
    return fit_histories

    



def eval_model(path):
    # print(path)
    fitness_array = np.zeros((pow(2, 16)), dtype=np.uint8)
    lower_bits = np.unpackbits(np.arange(256, dtype=np.uint8)).reshape((256, 8))
    for i in range(256):
        higher_bits = np.unpackbits(np.ones(256, dtype=np.uint8) * i).reshape((256, 8))
        full_bits = np.concatenate([higher_bits, lower_bits], axis=1)
        fitness = fitness_calculate(full_bits)
        fitness_array[i * 256: (i+1) * 256] = fitness

    if path == "greedy":
        sel = lambda metrics, p : metrics[:, 0]
    elif path == "mincrit":
        sel = lambda metrics, p: np.minimum(metrics[:, 0], 16)
    elif path == "novelty":
        sel = lambda metrics, p: metrics[:, 3]
    elif path == "localcomp":
        def sel(metrics, p):
            fits = metrics[:, 3]
            hammings = hammings_calc(p)
            for n in range(POP_SIZE * 2):
                smallests = np.argpartition(hammings[n], 5)[:5]
                local_fits = np.concatenate([[metrics[n, 0]], metrics[:, 0][smallests]])
                
                ranked = ss.rankdata(local_fits)
                norm = (ranked-1) / (len(ranked)-1)
                local_rank = norm[0]
                fits[n] += local_rank * 10
            return fits
                
            
    elif path == "random":
        sel = lambda metrics, p: metrics[:, 4]
    elif path == "mapelite":
        sel = "mapelite"
    elif 'linear' in path:
        genome_net = GroupGenomeLinear()
        genome_net.load_state_dict(torch.load(path))
        sel = lambda metrics, p: genome_net.forward(torch.tensor(metrics).float()).detach().numpy()
    else:
        genome_net = GroupGenome()
        genome_net.load_state_dict(torch.load(path))
        sel = lambda metrics, p: genome_net.forward(torch.tensor(metrics).float()).detach().numpy()
        
    def eval_worker(work_queue, result_queue, worker_id):
        np.random.seed(worker_id)
        while True:
            job_id = work_queue.get()
            if sel == "mapelite":
                run_log = map_elite()
                run_log = np.array(run_log)
                np.save("res_npy/"+str(DOMAIN)+"_mapelite_"+str(job_id)+".npy", run_log)
                result_queue.put(run_log[-1])
            else:
                run_log = run(sel, fitness_array)
                run_log = np.array(run_log)
                f_path = path if 'res/' not in path else path[4:]
                np.save("res_npy/"+str(DOMAIN)+"_"+f_path+"_"+str(job_id)+".npy", run_log)
                result_queue.put(run_log[-1, 0])
            
    work_queue = mp.Queue()
    result_queue = mp.Queue()
    processes = []
    for i in range(10):
        p = mp.Process(target=eval_worker, args=(work_queue, result_queue, i))
        p.start()
        processes.append(p)

    for k in range(20):
        work_queue.put(k)
    all_result = []
    for k in range(20):
        all_result.append(result_queue.get())
    for p in processes:
        p.terminate()
        p.join()
    # print(all_result)
    print(path + "|| Average: " + str(np.mean(all_result)))

if __name__ == "__main__":
    # DOMAIN = 0
    # evolve_group("convex_linear")
    # DOMAIN = 1
    # evolve_group("hashed_linear")
    # DOMAIN = 2
    # evolve_group("deceptive_linear")
    
    # DOMAIN = 0
    eval_model("res/convex1k.pt")
    eval_model("res/convex_linear1k.pt")
    eval_model("greedy")
    eval_model("mincrit")
    eval_model("novelty")
    eval_model("random")
    eval_model("localcomp")
    eval_model("mapelite")
    
    DOMAIN = 1
    eval_model("res/hashed1k.pt")
    eval_model("res/hashed_linear1k.pt")
    eval_model("greedy")
    eval_model("mincrit")
    eval_model("novelty")
    eval_model("random")
    eval_model("localcomp")
    eval_model("mapelite")
    
    DOMAIN = 2
    eval_model("res/deceptive1k.pt")
    eval_model("res/deceptive_linear1k.pt")
    eval_model("greedy")
    eval_model("mincrit")
    eval_model("novelty")
    eval_model("random")
    eval_model("localcomp")
    eval_model("mapelite")
    