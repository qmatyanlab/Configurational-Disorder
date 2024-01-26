import logging
import torch
import numpy as np
import io
import pickle as pkl

from pymatgen.core.periodic_table import Element

from torch_geometric.data import Data

def getElementProperties(number):
    element = Element.from_Z(number)
    properties = []
    properties.append(element.number)
    properties.append(float(element.atomic_mass))
    properties.append(element.atomic_radius)
    properties.append(element.electron_affinity)
    properties.append(element.row)
    properties.append(element.group)
    properties.append(element.data['X'])
    properties += list(element.data['Atomic orbitals'].values())[-6:-1]
    return properties

def calculateEdgeAttributes(dist, r_cutoff, dr):
    if dr == 0:
        return dist
    else:
        rgrid = np.arange(0, r_cutoff, dr)
        sigma = r_cutoff / 3
        attr = np.exp(-0.5 * (rgrid - dist)**2 /sigma**2) / np.sqrt(2 * np.pi) / sigma
        return attr

def structureToGraph(structure, atom_dict={k: k for k in range(100)}, r_cutoff=10, dr=0.1, max_neighbors=20):
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
    return data

def structureToData(structure, E, normalize_E=False, atom_dict={k: k for k in range(100)}, r_cutoff=10, dr=0.1, max_neighbors=20):
    graph = structureToGraph(structure, atom_dict=atom_dict, r_cutoff=r_cutoff, dr=dr, max_neighbors=max_neighbors)
    E = E / structure.num_sites if normalize_E else E
    graph.y = torch.tensor(E, dtype=torch.float)
    return graph

def get_logger(name):
    logger = logging.getLogger(name)
    filename = name + '.log'
    fh = logging.FileHandler(filename, mode='w')
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    return logger

class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)