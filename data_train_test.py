import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Data Loader

def get_data(Ai, ij):
    data = []
    for item in ij:
        feature = np.array([Ai[0][item[0]], Ai[0][item[1]]])
        for dim in range(1, Ai.shape[0]):
            feature = np.concatenate((feature, np.array([Ai[dim][item[0]], Ai[dim][item[1]]])))
        data.append(feature)
    return np.array(data)

class myDataset(Dataset):
    def __init__(self, dataset, fold, dimension, mode='train') -> None:
        super().__init__()

        self.mode = mode
        Ai = []
        A = np.load('data/ours/' + dataset + '/A_' + str(fold) + '.npy')
        Ai.append(A)
        for i in range(dimension - 1):
            tmp = np.dot(Ai[i], A)
            np.fill_diagonal(tmp, 0)
            tmp = tmp / np.max(tmp)
            Ai.append(copy.copy(tmp))
        Ai = np.array(Ai)
        positive_ij = np.load('data/ours/' + dataset + '/positive_ij.npy')
        negative_ij = np.load('data/ours/' + dataset + '/negative_ij.npy')
        positive5foldsidx = np.load('data/ours/' + dataset + '/positive5foldsidx.npy', allow_pickle=True)
        negative5foldsidx = np.load('data/ours/' + dataset + '/negative5foldsidx.npy', allow_pickle=True)

        if mode == 'test':
            positive_test_ij = positive_ij[positive5foldsidx[fold]['test']]
            negative_test_ij = negative_ij[negative5foldsidx[fold]['test']]
            positive_test_data = torch.Tensor(get_data(Ai, positive_test_ij))
            negative_test_data = torch.Tensor(get_data(Ai, negative_test_ij))
            self.data = torch.cat((positive_test_data, negative_test_data)).transpose(1, 2)
            self.target = torch.Tensor([1] * len(positive_test_ij) + [0] * len(negative_test_ij))

        elif mode == 'train':
            positive_train_ij = positive_ij[positive5foldsidx[fold]['train']]
            negative_train_ij = negative_ij[negative5foldsidx[fold]['train']]
            positive_train_data = torch.Tensor(get_data(Ai, positive_train_ij))
            negative_train_data = torch.Tensor(get_data(Ai, negative_train_ij))
            self.data = torch.cat((positive_train_data, negative_train_data)).transpose(1, 2)
            self.target = torch.Tensor([1] * len(positive_train_ij) + [0] * len(negative_train_ij))

        print('Finished reading the {} set of Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.data.shape[1:]))

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.data[index], self.target[index]
        else:
            return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)

def prep_dataloader(dataset, fold, mode, dimension, batch_size, n_jobs=0):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = myDataset(dataset, fold, dimension, mode=mode)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode=='train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    return dataloader

# Train

def train(tr_set, model, config, gpu):
    
    criterion = nn.BCELoss()
    n_epochs = config['n_epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    loss_record = []
    epoch = 0
    while epoch < n_epochs:
        model.train()
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(gpu), y.to(gpu)
            pred = model(x)
            bce_loss = criterion(pred, y)
            bce_loss.backward()
            optimizer.step()
            loss_record.append(bce_loss.detach().cpu().item())
        epoch += 1

        if bce_loss > 1:
            break

        print(epoch, bce_loss.detach().cpu().item())
        
    print('Finished training after {} epochs'.format(epoch))
    return loss_record

# Test

def test(tt_set, model, device):
    model.eval()
    preds = []
    labels = []
    for x, y in tt_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
        labels.append(y.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return labels, preds