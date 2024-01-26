import pickle as pkl
import matplotlib.pyplot as plt
import optuna

import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from model.model_embedding import CGNN, GAT, Transformer, MPNN
import utilities
from dataset import Dataset

class GNN():
    def __init__(self, dataset_root, modelname, num_hidden_layers, num_hidden_channels, num_heads,
                 lr=0.1, weight_decay=5e-4, batchsz=128, max_epoch=200, loss_type='mean'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = self.loadData(root=dataset_root, batchsz=batchsz)

        if modelname == 'CGNN' or modelname == 'MPNN':
            self.model = eval(modelname)(num_hidden_layers=num_hidden_layers, num_hidden_channels=num_hidden_channels,
                              num_edge_features=self.data[0].dataset.num_edge_features, device=self.device)
            self.name = modelname + '_' + str(num_hidden_layers) + '_' + str(num_hidden_channels) + '_' \
                        + str(lr) + '_' + str(weight_decay) + '_' + str(batchsz)
        elif modelname == 'GAT' or modelname == 'Transformer':
            self.model = eval(modelname)(num_hidden_layers=num_hidden_layers, num_hidden_channels=num_hidden_channels, num_heads=num_heads,
                             num_edge_features=self.data[0].dataset.num_edge_features, device=self.device)
            self.name = modelname + '_' + str(num_hidden_layers) + '_' + str(num_hidden_channels) + '_' + str(num_heads) + '_' \
                        + str(lr) + '_' + str(weight_decay) + '_' + str(batchsz)
        self.model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.max_epoch = max_epoch
        self.logger = utilities.get_logger('./save/' + self.name)
        self.loss_type = loss_type

        self.train_loss = []
        self.val_loss = []
        self.min_val_loss = float('inf')
        self.early_schedule_step = 0
        self.test_loss = float('inf')

    def loadData(self, root, batchsz=128, train_ratio=0.6, val_ratio=0.2):
        dataset = Dataset(root)
        dataset = dataset.shuffle()

        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = int(total_size * (1 - train_ratio - val_ratio))
        train_loader = DataLoader(dataset[: train_size], batch_size=batchsz, shuffle=True)
        val_loader = DataLoader(dataset[-(val_size + test_size):-test_size], batch_size=batchsz)
        test_loader = DataLoader(dataset[-test_size:], batch_size=batchsz)
        return [train_loader, val_loader, test_loader]

    def lossFunction(self, out, y):
        errors = out - y
        mse = torch.sum(torch.square(errors))
        error_mu = torch.mean(errors)
        var = torch.sum(torch.square(errors - error_mu))

        if self.loss_type == 'mse':
            return mse
        elif self.loss_type == 'variance':
            return var
        else:
            self.logger.info('Loss_type only supports mse and variance.')

    def saveGNNResults(self):
        results = {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'test_loss': self.test_loss
        }
        with open('./save/' + self.name + '.pkl', 'wb') as f:
            pkl.dump(results, f)

    def saveTrainedModel(self):
        with open('./save/' + self.name + '_model.pkl', 'wb') as f:
            pkl.dump(self.model, f)

    def saveTrainedModel(self):
        with open('./save/' + self.name + '_model.pkl', 'rb') as f:
            return pkl.load(f)

    def plotLossValues(self):
        epoch = range(len(self.train_loss))

        plt.plot(epoch, self.train_loss, 'k', label='train')
        plt.plot(epoch, self.val_loss, 'r', label='validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('./save/' + self.name + '.png')

    def trainModel(self, loader):
        self.model.train()
        train_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = self.lossFunction(out, batch.y)
            loss.backward()
            # clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            train_loss += loss.detach().cpu().numpy()
        train_loss = train_loss / len(loader.dataset)
        return train_loss

    def evalModel(self, loader):
        self.model.eval()
        eval_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch)
            loss = self.lossFunction(out, batch.y)
            eval_loss += loss.detach().cpu().numpy()
        eval_loss = eval_loss / len(loader.dataset)
        return eval_loss

    def run(self):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
        train_loader, val_loader, test_loader = self.data[0], self.data[1], self.data[2]

        for epoch in range(self.max_epoch):
            train_loss = self.trainModel(train_loader)
            val_loss = self.evalModel(val_loader)
            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)
            self.logger.info('epoch = {}, training loss = {}, validation loss = {}'.format(epoch, train_loss, val_loss))
            scheduler.step(train_loss)

            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                self.early_schedule_step = 0
                self.saveTrainedModel()
            else:
                self.early_schedule_step += 1
                self.logger.info('Early stopping step {}, the current validation loss {} is larger than best value {}'.
                                 format(self.early_schedule_step, val_loss, self.min_val_loss))

            # if self.early_schedule_step == 8:
            #     self.logger.info('Early stopped at epoch {}'.format(epoch))
            #     break

        self.model = self.loadTrainedModel()
        self.model.to(self.device)
        self.test_loss = self.evalModel(test_loader)
        self.logger.info('=' * 100)
        self.logger.info('The testing loss is {}'.format(self.test_loss))
        self.saveGNNResults()
        # self.plotLossValues()

class GNN_optuna():
    def __init__(self, dataset_root, loss_type='mean'):
        self.loss_type = loss_type
        self.dataset_root = dataset_root

    def save(self, study):
        with open('./save/' + self.loss_type + '.pkl', 'wb') as f:
            pkl.dump(study, f)

    def objective(self, trial):
        modelname = trial.suggest_categorical('modelname', ['CGNN', 'GAT', 'Transformer', 'MPNN'])
        num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 8)
        num_hidden_channels = trial.suggest_categorical('num_hidden_channels', [8, 16, 32, 64, 128])
        num_heads = trial.suggest_categorical('num_heads', [1, 2, 4, 8])
        lr = trial.suggest_float('lr', 1e-4, 1e-1)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1)
        batchsz = trial.suggest_int('batchsz', 2, 144)

        GNN_trial = GNN(dataset_root=self.dataset_root, modelname=modelname, num_hidden_layers=num_hidden_layers, num_hidden_channels=num_hidden_channels,
            num_heads=num_heads, lr=lr, weight_decay=weight_decay, batchsz=batchsz, loss_type=self.loss_type, max_epoch=100)
        GNN_trial.run()
        return GNN_trial.test_loss

    def run(self):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=10)

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items(): print("    {}: {}".format(key, value))

        self.save(study)

if __name__ == '__main__':
    torch_geometric.seed_everything(4)

    GNN_optuna = GNN_optuna(dataset_root='dataset', loss_type='mse')
    GNN_optuna.run()