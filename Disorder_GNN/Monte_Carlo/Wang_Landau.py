import pickle as pkl
import numpy as np
import copy
from tqdm import tqdm

import torch

from pymatgen.core import Structure

from utilities import (
generateSupercell, setConcentration, addRandomDistortion, structureToGraph,
get_logger, CPU_Unpickler
)

class WangLandau():
    def __init__(self, modelname=None, unitcell=None, concentration=0.5, flatness_criterion=0.8, steps_check_flatness=1000, max_steps=50000,
                 Emin=float('-inf'), Emax=float('inf'), dE=0.01, max_distortion=0, restart=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MC_name = 'WL_' + modelname + '_' + str(concentration)
        self.logger = get_logger('./save/' + self.MC_name)

        self.steps_check_flatness = steps_check_flatness
        self.flatness_criterion = flatness_criterion
        self.max_steps = max_steps

        if restart == None:
            start_configuration = copy.deepcopy(unitcell)
            generateSupercell(start_configuration, target_length=15)
            setConcentration(start_configuration, concentration)
            addRandomDistortion(start_configuration, max_distortion)
            self.graph_cur = structureToGraph(start_configuration)

            self.model = self.loadModel(modelname)

            self.E = (Emin, Emax, dE)
            self.Egrid = np.arange(Emin, Emax, dE)
            self.histogram = np.zeros(len(self.Egrid))
            self.lnDOS = np.zeros(len(self.Egrid))
            self.lnDOS_record = []
            self.lnf = 1
        else:
            with open(restart, 'rb') as f:
                WL = pkl.load(f)

            for key, value in WL.items():
                setattr(self, key, value)

    def loadModel(self, modelname):
        with open('./save/' + modelname + '.pkl', 'rb') as f:
            if self.device == torch.device('cuda'):
                model = pkl.load(f)
            elif self.device == torch.device('cpu'):
                model = CPU_Unpickler(f).load()
        return model.to(self.device)

    def evaluateEnergy(self, graph):
        graph.to(self.device)
        with torch.no_grad():
            energy = self.model(graph).cpu().numpy() * len(graph.x)
        return energy

    def getIndexByEnergy(self, energy):
        return int((energy - self.E[0]) // self.E[2])

    def getEnergyByIndex(self, index):
        return index * self.E[2] + self.E[0]

    def swap(self, graph):
        elem_list = np.array(graph.x)
        elem_kind = list(set(elem_list))
        index0 = np.random.choice(np.where(elem_list == elem_kind[0])[0])
        index1 = np.random.choice(np.where(elem_list == elem_kind[1])[0])
        graph.x[index0] = torch.tensor(elem_kind[1])
        graph.x[index1] = torch.tensor(elem_kind[0])

    def accept(self, energy, energy_new):
        index = self.getIndexByEnergy(energy)
        index_new = self.getIndexByEnergy(energy_new)
        lnDOS = self.lnDOS[index]
        lnDOS_new = self.lnDOS[index_new]
        if lnDOS >= lnDOS_new:
            return True
        return np.exp(lnDOS - lnDOS_new) > np.random.uniform(0, 1)

    def runMC(self):
        energy_cur = self.evaluateEnergy(self.graph_cur)

        for step_counter in tqdm(range(1, self.max_steps)):
            graph_next = self.graph_cur.clone()
            self.swap(graph_next)
            energy_next = self.evaluateEnergy(graph_next)

            if self.accept(energy_cur, energy_next):
                energy_cur = energy_next
                self.graph_cur = graph_next

            index = self.getIndexByEnergy(energy_cur)
            self.lnDOS[index] += self.lnf
            self.histogram[index] += 1

            if step_counter % self.steps_check_flatness == 0:
                histogram = self.histogram[self.lnDOS > 0]
                print('histogram min = {}, max = {}, mean = {}.'.format(histogram.min(), histogram.max(), histogram.mean()))
                if len(histogram) >= 2 and (histogram > self.flatness_criterion * histogram.mean()).all():
                    self.histogram[:] = 0
                    self.lnf = self.lnf / 2
                    print('the flatness criterion is satisfied. The current modification factor is exp({})'.format(self.lnf))

            if step_counter % 1 == 0:
                self.lnDOS_record.append(self.lnDOS.copy())

        results = {
            'graph_cur': self.graph_cur,
            'model': self.model,
            'E': self.E,
            'Egrid': self.Egrid,
            'histogram': self.histogram,
            'lnDOS': self.lnDOS,
            'lnDOS_record': self.lnDOS_record,
            'lnf': self.lnf
        }
        with open('./save/'+ self.MC_name + '_results.pkl', 'wb') as f:
            pkl.dump(results, f)

if __name__ == '__main__':
    structure = Structure.from_file('unitcell.vasp')
    modelname = 'distortion_best_model'
    concentration = 0.5

    Emin = -830
    Emax = -790
    dE = (Emax - Emin) / 300
    WL = WangLandau(modelname=modelname,
                    unitcell=structure,
                    concentration=concentration,
                    max_steps=int(2.5E5 * 200),
                    Emin=Emin,
                    Emax=Emax,
                    dE=dE,
                    max_distortion=0.2,
                    restart=None
                    )
    WL.runMC()

