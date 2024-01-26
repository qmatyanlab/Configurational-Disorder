import os
import pandas as pd
import numpy as np
import random
import json

import torch
from torch_geometric.data import InMemoryDataset

from pymatgen.core import Structure

import utilities

class Dataset(InMemoryDataset):
    def __init__(self, root, dr=0.1, normalize_E=True, transform=None, pre_transform=None):
        self.dr = dr
        self.normalize_E = normalize_E
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def raw_file_names(self):
        return []

    def processed_file_names(self):
        return ['dataset.pt']

    def download(self): pass

    def process(self):
        df = self.getDFTEnergy()
        atom_dict = self.getAtomFeatures()

        raw_dir = os.path.join(self.root, 'raw')

        data_list = []
        for ind, E in zip(df.iloc[:, 0], df.iloc[:, 1]):
            structure = Structure.from_file(raw_dir + '/' + str(ind) + '.POSCAR')
            data = utilities.structureToData(structure=structure, E=E, normalize_E=self.normalize_E,
                                             atom_dict=atom_dict, dr=self.dr)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def getDFTEnergy(self):
        raw_files = os.listdir(os.path.join(self.root, 'raw'))
        assert 'energy.dat' in raw_files, 'Energy file doesn not exist.'
        df = pd.read_csv(self.raw_dir + '/energy.dat', delimiter=' ', names=['index', 'energy'],
                         dtype={'index': int, 'energy': float})
        df = df.sort_values('index')
        return df

    def getAtomFeatures(self):
        raw_files = os.listdir(self.raw_dir)
        if 'atom_init.json' in raw_files:
            with open(self.raw_dir + '/atom_init.json') as json_file:
                atom_dict = json.load(json_file)
            atom_dict = {int(k): v for k, v in atom_dict.items()}
        else:
            atom_dict = {k: k for k in range(100)}
        return atom_dict

if __name__ == '__main__':
    data = Dataset(root='dataset')