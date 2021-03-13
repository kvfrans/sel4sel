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

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.color_palette()

BIT_LENGTH = 16
HALF_LENGTH = 8
LAYERS = 4
MUTATION_RATE = 0.05
POP_SIZE = 50
GENERATIONS = 2000

# Deceptive Bits.
def fitness_calculate(bit_slice):
    total_fitness = np.zeros(bit_slice.shape[0])
    for n in range(LAYERS):
        layer_iter = pow(2, n)
        layer_len = pow(2, LAYERS-n)
        for j in range(0, layer_len, 2):
            equals = np.not_equal(bit_slice[:, (j+0)*layer_iter:(j+1)*layer_iter], bit_slice[:, (j+1)*layer_iter:(j+2)*layer_iter])
            total_match = np.sum(equals, axis=1)
            total_fitness += np.equal(total_match, 0) * layer_iter
    return total_fitness

# Convex Bits.
# def fitness_calculate(bit_slice):
#     total_fitness = np.sum(bit_slice, axis=1) * 2
#     return total_fitness

# Hashed Bits.
# def fitness_calculate(bit_slice):
#     random_vals = np.clip(np.abs(np.random.normal(loc=0, scale=(4/3), size=(bit_slice.shape[0],))), 0, 5)
#     total_fitness = np.power(2, random_vals)
#     return total_fitness

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

def run(sel_func, fitness_array):
    log = []

    # Instantiate random population.
    population = np.stack([np.random.choice([True, False], size=(BIT_LENGTH))] * POP_SIZE*2, axis=0)
    population_metrics = np.zeros((POP_SIZE*2, 6), dtype=np.float)
    population_metrics[:POP_SIZE, 0] = fitness_calculate_fast(population[:POP_SIZE], fitness_array)

    for i in range(GENERATIONS):

        # Increase age
        population_metrics[:POP_SIZE, 2] += 1

        # Make children
        population[POP_SIZE:] = mutate(population[:POP_SIZE])
        # population[POP_SIZE:] = np.random.choice([True, False], size=(POP_SIZE, BIT_LENGTH))
        population_metrics[POP_SIZE:, 0] = fitness_calculate_fast(population[POP_SIZE:], fitness_array)

        # Compute metrics for every individual.
        compute_metrics(population_metrics, population, i)

        # Compute internal fitness with selection function
        sel = sel_func(population_metrics)

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
        start = time.time()
        genome_net.set_parameters(params)
        genome_sel = lambda metrics : genome_net.forward(torch.tensor(metrics).float()).numpy()
        run_log = run(genome_sel, fitness_array)
        gfit = run_log[GENERATIONS-1][0]
        result_queue.put((params, gfit))
        # print(time.time() - start)

def evolve_group():
    genome_net = GroupGenome()
    genome_queue = mp.Queue()
    result_queue = mp.Queue()
    processes = []
    for i in range(10):
        p = mp.Process(target=worker, args=(genome_queue, result_queue, i))
        p.start()
        processes.append(p)


    for i in range(10000):
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
        torch.save(genome_net.state_dict(), "res/" + sys.argv[1] + ".pt")

def eval_model(path):
    fitness_array = np.zeros((pow(2, 16)), dtype=np.uint8)
    lower_bits = np.unpackbits(np.arange(256, dtype=np.uint8)).reshape((256, 8))
    for i in range(256):
        higher_bits = np.unpackbits(np.ones(256, dtype=np.uint8) * i).reshape((256, 8))
        full_bits = np.concatenate([higher_bits, lower_bits], axis=1)
        fitness = fitness_calculate(full_bits)
        fitness_array[i * 256: (i+1) * 256] = fitness
        
    genome_net = GradientGenome()
    genome_net.load_state_dict(torch.load(path))
    genome_sel = lambda metrics: genome_net.forward(torch.tensor(metrics).float()).detach().numpy()

    for k in range(20):
        run_log = run(genome_sel, fitness_array)
        run_log = np.array(run_log)
        np.save("data/hiff_learned_{}.npy".format(k), run_log)
        print(run_log[-5:, 0].tolist())


if __name__ == "__main__":
    # To train a model. Make sure to uncomment the domain you want to train on in lines 26-46.
    evolve_group()

    # Run this to test the performance of a learned model.
    # eval_model("res_remote/hiff.pt")
