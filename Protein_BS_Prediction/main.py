import pickle
import sys
import timeit
import scipy
import numpy as np
from math import sqrt
import scipy
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
from data_merge import data_load
from network.model import Representation_model
torch.multiprocessing.set_start_method('spawn')
from metric import *

def pack(prot_features, labels, device):
    N = len(prot_features)
    Lmax = 0
    for label in labels:
        if label.shape[0] >= Lmax:
            Lmax = label.shape[0]

    prot_feature_new = torch.zeros((N, Lmax, 1024), device=device)
    i = 0
    for prot_feature in prot_features:
        prot_feature_len = prot_feature.shape[0]
        prot_feature_new[i, 0:prot_feature_len-1, :] = torch.tensor(prot_feature[:-1]).to(device)
        i += 1

    label_new = torch.zeros((N, Lmax), dtype=torch.long, device=device)
    i = 0
    for label in labels:
        label_len = label.shape[0]
        label_new[i, 0:label_len] = label
        i += 1

    return prot_feature_new, label_new


# class Trainer(object):
#     def __init__(self, model, batch_size, lr, weight_decay):
#         self.model = model
#         self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
#         # self.schedule = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150, eta_min=0)
#         self.batch_size = batch_size
#         # self.optimizer = Ranger(self.model.parameters(), lr=lr, weight_decay=weight_decay)
#
#     def train(self, dataset, epoch, device):
#         np.random.shuffle(dataset)
#         N = len(dataset)
#         loss_total = 0
#         i = 0
#         # self.optimizer = torch.nn.DataParallel(self.optimizer, device_ids=[0, 1])
#         self.optimizer.zero_grad()
#
#         for data in dataset:
#             prot_feature, label = data
#             prot_features, labels = prot_feature[:-1].unsqueeze(0).to(device), label.unsqueeze(0).to(device)
#             data = (prot_features, labels)
#             loss = self.model(data, device, task="BS prediction", train=True)#.mean()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=8)
#             self.optimizer.step()
#             self.optimizer.zero_grad()
#             i = i + 1
#             loss_total += loss.item()
#         return loss_total
#
# class Tester(object):
#     def __init__(self, model, batch_size):
#         self.model = model
#         self.batch_size = batch_size
#
#     def test(self, dataset, device):
#         N = len(dataset)
#         pred_bs_collect, label_collect = [], []
#         i = 0
#         prot_features, labels = [], []
#         for data in dataset:
#             prot_feature, label = data
#             prot_features, labels = prot_feature[:-1].unsqueeze(0).to(device), label.unsqueeze(0).to(device)
#             data = (prot_features, labels)
#             pred_bs, label = self.model(data, device, task="BS prediction", train=False)
#             pred_bs = pred_bs.to('cpu').data.numpy()
#             label = label.to('cpu').data.numpy()
#             pred_bs_collect.extend(pred_bs)
#             label_collect.extend(label)
#             i = i + 1
#         return caculate_metric(pred_bs_collect, label_collect)
#
#     def save_AUCs(self, AUCs, filename):
#         with open(filename, 'a') as f:
#             f.write('\t'.join(map(str, AUCs)) + '\n')
#
#     def save_model(self, model, filename):
#         torch.save(model.state_dict(), filename)

class Trainer(object):
    def __init__(self, model, batch_size, lr, weight_decay):
        self.model = model
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        # self.schedule = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150, eta_min=0)
        self.batch_size = batch_size
        # self.optimizer = Ranger(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, dataset, epoch, device):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        i = 0
        # self.optimizer = torch.nn.DataParallel(self.optimizer, device_ids=[0, 1])
        self.optimizer.zero_grad()

        prot_features, labels = [], []
        for data in dataset:
            prot_feature, label = data
            i = i + 1
            prot_features.append(prot_feature)
            labels.append(label)
            if i % self.batch_size == 0 or i == N:
                if len(prot_features) != 1:
                    prot_features, labels = pack(prot_features, labels, device)
                    data = (prot_features, labels)
                    loss = self.model(data, device, task="BS prediction", train=True)#.mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=8)
                    prot_features, labels = [], []
                else:
                    prot_features, labels = [], []
            else:
                continue

            if i % self.batch_size == 0 or i == N:
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_total += loss.item()
        return loss_total

class Tester(object):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def test(self, dataset, device):
        N = len(dataset)
        pred_bs_collect, label_collect = [], []
        i = 0
        prot_features, labels = [], []
        for data in dataset:
            prot_feature, label = data
            i = i + 1
            prot_features.append(prot_feature)
            labels.append(label)

            if i % self.batch_size == 0 or i == N:
                prot_features, labels = pack(prot_features, labels, device)
                data = (prot_features, labels)
                pred_bs, label = self.model(data, device, task="BS prediction", train=False)
                pred_bs = pred_bs.to('cpu').data.numpy()
                label = label.to('cpu').data.numpy()
                pred_bs_collect.extend(pred_bs)
                label_collect.extend(label)
                prot_features, labels = [], []
            else:
                continue

        return caculate_metric(pred_bs_collect, label_collect)

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    import os
    dataset_name = "DNA"
    train_name = "DNA-646"
    test_name = "DNA-181"
    result_ID = "3"
    iteration = 20
    decay_interval = 3
    batchs = 2
    lr = 1e-4
    weight_decay = 0.2
    lr_decay = 0.5
    setting = dataset_name + "_" + train_name + "_" + test_name + "_" + result_ID
    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    dataset_train, dataset_test = data_load(dataset_name, train_name, test_name, device)
    setup_seed(2023)
    model = Representation_model(3, 512, 256, 64, 128, batchs, device).to(device)
    target_model_state_dict = model.state_dict()
    source_model_state_dict = torch.load("output/model/pretrain_new_2")
    # print(source_model_state_dict.keys())
    partial_dict = {
        "represent.weight": source_model_state_dict["represent.weight"],
        "represent.bias": source_model_state_dict["represent.bias"],
        "atten.0.q_layers.weight": source_model_state_dict["atten.0.q_layers.weight"],
        "atten.0.q_layers.bias": source_model_state_dict["atten.0.q_layers.bias"],
        "atten.0.k_layers.weight": source_model_state_dict["atten.0.k_layers.weight"],
        "atten.0.k_layers.bias": source_model_state_dict["atten.0.k_layers.bias"],
        "atten.0.v_layers.weight": source_model_state_dict["atten.0.v_layers.weight"],
        "atten.0.v_layers.bias": source_model_state_dict["atten.0.v_layers.bias"],
        "atten.1.q_layers.weight": source_model_state_dict["atten.1.q_layers.weight"],
        "atten.1.q_layers.bias": source_model_state_dict["atten.1.q_layers.bias"],
        "atten.1.k_layers.weight": source_model_state_dict["atten.1.k_layers.weight"],
        "atten.1.k_layers.bias": source_model_state_dict["atten.1.k_layers.bias"],
        "atten.1.v_layers.weight": source_model_state_dict["atten.1.v_layers.weight"],
        "atten.1.v_layers.bias": source_model_state_dict["atten.1.v_layers.bias"],
        "atten.2.q_layers.weight": source_model_state_dict["atten.2.q_layers.weight"],
        "atten.2.q_layers.bias": source_model_state_dict["atten.2.q_layers.bias"],
        "atten.2.k_layers.weight": source_model_state_dict["atten.2.k_layers.weight"],
        "atten.2.k_layers.bias": source_model_state_dict["atten.2.k_layers.bias"],
        "atten.2.v_layers.weight": source_model_state_dict["atten.2.v_layers.weight"],
        "atten.2.v_layers.bias": source_model_state_dict["atten.2.v_layers.bias"],
        "norms1.0.weight": source_model_state_dict["norms1.0.weight"],
        "norms1.0.bias": source_model_state_dict["norms1.0.bias"],
        "norms1.1.weight": source_model_state_dict["norms1.1.weight"],
        "norms1.1.bias": source_model_state_dict["norms1.1.bias"],
        "norms1.2.weight": source_model_state_dict["norms1.2.weight"],
        "norms1.2.bias": source_model_state_dict["norms1.2.bias"],
        "decoder.linear_in.weight": source_model_state_dict["decoder.linear_in.weight"],
        "decoder.linear.0.weight": source_model_state_dict["decoder.linear.0.weight"],
        "decoder.linear.0.bias": source_model_state_dict["decoder.linear.0.bias"],
        "decoder.linear.1.weight": source_model_state_dict["decoder.linear.1.weight"],
        "decoder.linear.1.bias": source_model_state_dict["decoder.linear.1.bias"],
        "decoder.linear_out.weight": source_model_state_dict["decoder.linear_out.weight"],
        "decoder.linear_out.bias": source_model_state_dict["decoder.linear_out.bias"]
    }
    target_model_state_dict.update(partial_dict)
    model.load_state_dict(target_model_state_dict)
    # model = torch.nn.DataParallel(model, device_ids=[1], output_device=1)
    # model = model.module.to(torch.device('cpu'))
    trainer = Trainer(model, batchs, lr, weight_decay)
    tester = Tester(model, batchs)

    """Output files."""
    file_AUCs = 'output/result/AUCs--' + setting + '.txt'
    file_model = 'output/model/' + setting
    AUCs = ('Epoch\tTime(sec)\tLoss_train\t'
            'Specificity\tRecall\tPrecision\tF1_score\tMCC\tAUC')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()
    auc1 = 0
    for epoch in range(1, iteration):

        if epoch % decay_interval == 0 and trainer.optimizer.param_groups[0]['lr'] > 2e-5:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train, epoch, device)
        Specificity, Recall, Precision, F1_score, MCC, AUC = tester.test(dataset_test, device)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train,
                Specificity, Recall, Precision, F1_score, MCC, AUC]
        tester.save_AUCs(AUCs, file_AUCs)
        # tester.save_model(model, file_model)
        print('\t'.join(map(str, AUCs)))
        if auc1 < AUC:
            auc1 = AUC
            tester.save_model(model, file_model)

        # "decoder.Attention.linear_layers.0.weight": source_model_state_dict["decoder.Attention.linear_layers.0.weight"],
        # "decoder.Attention.linear_layers.0.bias": source_model_state_dict["decoder.Attention.linear_layers.0.bias"],
        # "decoder.Attention.linear_layers.1.weight": source_model_state_dict["decoder.Attention.linear_layers.1.weight"],
        # "decoder.Attention.linear_layers.1.bias": source_model_state_dict["decoder.Attention.linear_layers.1.bias"],
        # "decoder.Attention.linear_layers.2.weight": source_model_state_dict["decoder.Attention.linear_layers.2.weight"],
        # "decoder.Attention.linear_layers.2.bias": source_model_state_dict["decoder.Attention.linear_layers.2.bias"],


# 6	230.8043248	6.830619469496014	0.9521743517304005	0.5284974093264249	0.5182926829268293	0.5233453052847614	0.47646543482157433	0.8995526507157223
# 3	105.6596771	74.42660399340093	0.9738341601700922	0.4714285714285714	0.5336028297119757	0.5005925574780753	0.47197250461171475	0.9221064910904121