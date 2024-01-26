import io
import random
import numpy as np
import pickle as pkl
import logging

import torch
from torch_geometric.data import Data

R_CUTOFF = 10
MAXNEIGHBORS = 20

def get_logger(name):
    logger = logging.getLogger(name)
    filename = name + '.log'
    fh = logging.FileHandler(filename, mode='w')
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    return logger

def calculateEdgeAttributes(dist, dr):
    if dr == 0:
        return dist
    else:
        rgrid = np.arange(0, R_CUTOFF, dr)
        sigma = R_CUTOFF / 3
        attr = np.exp(-0.5 * (rgrid - dist)**2 /sigma**2) / np.sqrt(2 * np.pi) / sigma
        return attr

def structureToGraph(structure, r_cutoff=10, dr=0.1, max_neighbors=20):
    atom_dict = {k: k for k in range(100)}
    neighbors = structure.get_all_neighbors(r_cutoff, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x.nn_distance) for nbrs in neighbors]
    neighbors_idx, neighbors_dist = [], []
    max_neighbors = min(max_neighbors, len(all_nbrs[0]))
    for nbr in all_nbrs:
        neighbors_idx.append(list(map(lambda x: x.index, nbr[ : max_neighbors])))
        neighbors_dist.append(list(map(lambda x: x.nn_distance, nbr[ : max_neighbors])))

    x = []
    edge_index = []
    edge_attr = []
    for i in range(len(structure.atomic_numbers)):
        elemi = atom_dict[structure.atomic_numbers[i]]
        x.append(elemi)
        for j in range(len(neighbors_idx[i])):
            edge_index.append([i, neighbors_idx[i][j]])
            edge_attr.append(calculateEdgeAttributes(neighbors_dist[i][j], r_cutoff=r_cutoff, dr=dr))

    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(np.array(edge_index), dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.batch = torch.zeros((data.x.shape[0]), dtype=int)
    return data

def generateSupercell(unitcell, target_length):
    l = np.array([unitcell.lattice.a, unitcell.lattice.b, unitcell.lattice.c])
    n = np.rint(target_length / l)
    unitcell.make_supercell(n)

def setConcentration(cell, concentration):
    Cu, Au = cell.types_of_species[0], cell.types_of_species[1]
    num_Au = int(cell.num_sites * concentration)
    indices = random.sample(range(cell.num_sites), num_Au)
    for j in range(cell.num_sites):
        if j in indices:
            cell[j].species = Au
        else:
            cell[j].species = Cu

def addRandomDistortion(structure, max_distortion):
    for j in range(len(structure)):
        displacement = np.random.random_sample(3) * max_distortion - max_distortion / 2
        structure[j].coords += displacement

class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)