import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, auc

from LDAfomer import LDAformer, CNN, NN
from data_train_test import prep_dataloader, train, test

# Cofig

gpu = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

config = {
    'dataset': 'dataset1',
    'fold': 0,
    'n_epochs': 30,
    'batch_size': 32,
    'dimension': 3,
    'd_model': 12,
    'n_heads': 1,
    'e_layers': 4,
    'd_ff': 0.5
}

# config = {
#     'dataset': 'dataset2',
#     'fold': 0,
#     'n_epochs': 10,
#     'batch_size': 32,
#     'dimension': 3,
#     'd_model': 6,
#     'n_heads': 3,
#     'e_layers': 2,
#     'd_ff': 1
# }

tr_set = prep_dataloader(dataset=config['dataset'], fold=config['fold'], mode='train', dimension=config['dimension'], batch_size=config['batch_size'])
tt_set = prep_dataloader(dataset=config['dataset'], fold=config['fold'], mode='test', dimension=config['dimension'], batch_size=config['batch_size'])

seq_len = tr_set.dataset[0][0].shape[0]
d_input = tr_set.dataset[0][0].shape[1]

while True:
    model = LDAformer(
        seq_len=seq_len, d_input=d_input, d_model=config['d_model'], 
        n_heads=config['n_heads'], d_ff=config['d_ff'], e_layers=config['e_layers'], 
        pos_emb=False, value_sqrt=False
    ).to(gpu)
    model_loss_record = train(tr_set, model, config, gpu)
    if model_loss_record[-1] < 1:
        break

labels, preds = test(tt_set, model, gpu)

AUC = roc_auc_score(labels, preds)
precision, recall, _ = precision_recall_curve(labels, preds)
AUPR = auc(recall, precision)
preds = np.array([1 if p > 0.5 else 0 for p in preds])
ACC = accuracy_score(labels, preds)
P = precision_score(labels, preds)
R = recall_score(labels, preds)
F1 = f1_score(labels, preds)

print(AUC, AUPR, ACC, P, R, F1)
