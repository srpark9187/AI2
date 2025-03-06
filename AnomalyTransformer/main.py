#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment


win_size = 100
input_c = 25
output_c = 25
lr = 1e-4
batch_size = 512
data_path = "./dataset/PSM"
anormly_ratio = 4.00

model = AnomalyTransformer(win_size= win_size, enc_in=input_c, c_out=output_c, e_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if torch.cuda.is_available():
    model.cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = "PSM"
model_save_path = "checkpoints"
model.load_state_dict(
    torch.load(
        os.path.join(str(model_save_path), str(dataset) + '_checkpoint.pth')))
model.eval()
#%%
batch_size = 512
data_path = "./dataset/PSM"
print("======================TEST MODE======================")
criterion = nn.MSELoss(reduce=False)
temperature = 50
# (1) stastic on the train set
def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

train_loader = get_loader_segment(data_path, batch_size=batch_size, win_size=win_size,
                                               mode='train',
                                               dataset=dataset)

#%%
attens_energy = []
for i, (input_data, labels) in enumerate(train_loader):
    input = input_data.float().to(device)
    output, series, prior, _ = model(input)
    loss = torch.mean(criterion(input, output), dim=-1)
    series_loss = 0.0
    prior_loss = 0.0
    for u in range(len(prior)):
        if u == 0:
            series_loss = my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)).detach()) * temperature
            prior_loss = my_kl_loss(
                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                        win_size)),
                series[u].detach()) * temperature
        else:
            series_loss += my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)).detach()) * temperature
            prior_loss += my_kl_loss(
                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                        win_size)),
                series[u].detach()) * temperature

    metric = torch.softmax((-series_loss - prior_loss), dim=-1)
    cri = metric * loss
    cri = cri.detach().cpu().numpy()
    attens_energy.append(cri)

attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
train_energy = np.array(attens_energy)
#%%

# (2) find the threshold
thre_loader = get_loader_segment(data_path, batch_size=batch_size, win_size=win_size,
                                        mode='thre',
                                        dataset=dataset)


anormly_ratio = 4.00
attens_energy = []
for i, (input_data, labels) in enumerate(thre_loader):
    input = input_data.float().to(device)
    output, series, prior, _ = model(input)
    loss = torch.mean(criterion(input, output), dim=-1)

    series_loss = 0.0
    prior_loss = 0.0
    for u in range(len(prior)):
        if u == 0:
            series_loss = my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)).detach()) * temperature
            prior_loss = my_kl_loss(
                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                        win_size)),
                series[u].detach()) * temperature
        else:
            series_loss += my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)).detach()) * temperature
            prior_loss += my_kl_loss(
                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                        win_size)),
                series[u].detach()) * temperature
    # Metric
    metric = torch.softmax((-series_loss - prior_loss), dim=-1)
    cri = metric * loss
    cri = cri.detach().cpu().numpy()
    attens_energy.append(cri)

attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
test_energy = np.array(attens_energy)
combined_energy = np.concatenate([train_energy, test_energy], axis=0)
thresh = np.percentile(combined_energy, 100 - anormly_ratio)
print("Threshold :", thresh)

#%%
#Threshold : 5.50419344537241e-22
thresh = 5.50419344537241e-22

# (3) evaluation on the test set
test_labels = []
attens_energy = []
for i, (input_data, labels) in enumerate(thre_loader):
    input = input_data.float().to(device)
    output, series, prior, _ = model(input)

    loss = torch.mean(criterion(input, output), dim=-1)

    series_loss = 0.0
    prior_loss = 0.0
    for u in range(len(prior)):
        if u == 0:
            series_loss = my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)).detach()) * temperature
            prior_loss = my_kl_loss(
                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                        win_size)),
                series[u].detach()) * temperature
        else:
            series_loss += my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)).detach()) * temperature
            prior_loss += my_kl_loss(
                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                        win_size)),
                series[u].detach()) * temperature
    metric = torch.softmax((-series_loss - prior_loss), dim=-1)

    cri = metric * loss
    cri = cri.detach().cpu().numpy()
    attens_energy.append(cri)
    test_labels.append(labels)

attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
test_energy = np.array(attens_energy)
test_labels = np.array(test_labels)

pred = (test_energy > thresh).astype(int)

gt = test_labels.astype(int)

print("pred:   ", pred.shape)
print("gt:     ", gt.shape)
#%%
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(gt, pred)
precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                        average='binary')
print(
    "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        accuracy, precision,
        recall, f_score))

accuracy, precision, recall, f_score

#%%
# detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
anomaly_state = False
for i in range(len(gt)):
    if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
        anomaly_state = True
        for j in range(i, 0, -1):
            if gt[j] == 0:
                break
            else:
                if pred[j] == 0:
                    pred[j] = 1
        for j in range(i, len(gt)):
            if gt[j] == 0:
                break
            else:
                if pred[j] == 0:
                    pred[j] = 1
    elif gt[i] == 0:
        anomaly_state = False
    if anomaly_state:
        pred[i] = 1

pred = np.array(pred)
gt = np.array(gt)
print("pred: ", pred.shape)
print("gt:   ", gt.shape)

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(gt, pred)
precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                        average='binary')
print(
    "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        accuracy, precision,
        recall, f_score))

accuracy, precision, recall, f_score

