import scipy.sparse as sp
import pickle
import sys
import timeit
import scipy
import numpy as np
from math import sqrt
import scipy
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
from data_merge import data_load
from network.model import Representation_model
from sklearn.metrics import mean_squared_error
torch.multiprocessing.set_start_method('spawn')


def pack(molecule_atoms, molecule_adjs, sequences, smiles, labels, p_LMs, d_LMs, device, sources=None):
    proteins_len = 1200
    words_len = 100
    atoms_len = 0
    p_l = 1200
    d_l = 100
    N = len(molecule_atoms)
    atom_num = []
    for atom in molecule_atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    molecule_atoms_new = torch.zeros((N, atoms_len, 75), device=device)
    i = 0
    for atom in molecule_atoms:
        a_len = atom.shape[0]
        molecule_atoms_new[i, :a_len, :] = atom
        i += 1

    molecule_adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in molecule_adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len, device=device)
        molecule_adjs_new[i, :a_len, :a_len] = adj
        i += 1

    protein_LMs = []
    molecule_LMs = []
    for sequence in sequences:
        protein_LMs.append(p_LMs[sequence])

    for smile in smiles:
        molecule_LMs.append(d_LMs[smile])

    protein_LM = torch.zeros((N, p_l, 1024), device=device)
    molecule_LM = torch.zeros((N, d_l, 768), device=device)
    # print(d_l)
    for i in range(N):
        C_L = molecule_LMs[i].shape[0]
        if C_L >= 100:
            molecule_LM[i, :, :] = torch.tensor(molecule_LMs[i][0:100, :]).to(device)
        else:
            molecule_LM[i, :C_L, :] = torch.tensor(molecule_LMs[i]).to(device)
        P_L = protein_LMs[i].shape[0]

        if P_L >= 1200:
            protein_LM[i, :, :] = torch.tensor(protein_LMs[i][0:1200, :]).to(device)
        else:
            protein_LM[i, :P_L, :] = torch.tensor(protein_LMs[i]).to(device)

    labels_new = torch.zeros(N, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    if sources != None:
        sources_new = torch.zeros(N, device=device)
        i = 0
        for source in sources:
            sources_new[i] = source
            i += 1
    else:
        sources_new = torch.zeros(N, device=device)

    return molecule_atoms_new, molecule_adjs_new, protein_LM, molecule_LM, labels_new, sources_new


class Trainer(object):
    def __init__(self, model, batch_size, lr, weight_decay):
        self.model = model
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                     weight_decay=weight_decay)
        # self.schedule = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150, eta_min=0)
        self.batch_size = batch_size
        # self.optimizer = Ranger(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, dataset, p_LMs, d_LMs):
        # np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        i = 0
        self.optimizer.zero_grad()

        molecule_atoms, molecule_adjs, sequences, smiles, labels, sources = [], [], [], [], [], []
        for data in dataset:
            molecule_atom, molecule_adj, sequence, smile, label, source = data
            if np.isnan(np.mean(d_LMs[smile])):
                continue
            i = i + 1
            molecule_atoms.append(molecule_atom)
            molecule_adjs.append(molecule_adj)
            sequences.append(sequence)
            smiles.append(smile)
            labels.append(label)
            sources.append(source)

            if i % self.batch_size == 0 or i == N:
                if len(molecule_atoms) != 1:
                    molecule_atoms, molecule_adjs, protein_LM, molecule_LM, labels, sources = pack(molecule_atoms,
                                                                                                   molecule_adjs,
                                                                                                   sequences, smiles,
                                                                                                   labels, p_LMs, d_LMs,
                                                                                                   device, sources)
                    data = (molecule_atoms, molecule_adjs, protein_LM, molecule_LM, labels, sources)
                    loss = self.model(data, device, "DTA_prediction")  # .mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=8)
                    molecule_atoms, molecule_adjs, sequences, smiles, labels, sources = [], [], [], [], [], []
                else:
                    molecule_atoms, molecule_adjs, sequences, smiles, labels, sources = [], [], [], [], [], []
            else:
                continue

            if i % self.batch_size == 0 or i == N:
                self.optimizer.step()
                # self.schedule.step()
                self.optimizer.zero_grad()
            loss_total += loss.item()
        return loss_total

def get_cindex(Y, P):
    summ = 0
    pair = 0
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i != j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * int(P[i] > P[j]) + 0.5 * int(P[i] == P[j])
    if pair != 0:
        return summ / pair
    else:
        return 0
class Tester(object):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size
    def test(self, dataset, p_LMs, d_LMs):
        N = len(dataset)
        T, S = [], []
        i = 0
        molecule_atoms, molecule_adjs, sequences, smiles, labels = [], [], [], [], []
        for data in dataset:
            molecule_atom, molecule_adj, sequence, smile, label = data
            if np.isnan(np.mean(d_LMs[smile])):
                continue
            i = i + 1
            molecule_atoms.append(molecule_atom)
            molecule_adjs.append(molecule_adj)
            sequences.append(sequence)
            smiles.append(smile)
            labels.append(label)

            if i % self.batch_size == 0 or i == N:
                molecule_atoms, molecule_adjs, protein_LM, molecule_LM, labels, _ = pack(molecule_atoms, molecule_adjs,
                                                                                         sequences, smiles, labels,
                                                                                         p_LMs, d_LMs,
                                                                                         device)

                data = (molecule_atoms, molecule_adjs, protein_LM, molecule_LM, labels, _)
                correct_labels, predicted_scores = self.model(data, device, "DTA_prediction", train=False)
                correct_labels = correct_labels.to('cpu').data.numpy()
                predicted_scores = predicted_scores.to('cpu').data.numpy()

                for j in range(len(correct_labels)):
                    T.append(correct_labels[j])
                    S.append(predicted_scores[j])

                molecule_atoms, molecule_adjs, sequences, smiles, labels = [], [], [], [], []
            else:
                continue

        CI = get_cindex(T, S)
        MSE = mean_squared_error(T, S)
        return CI, MSE
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

def calculate_PRC(T, S):
    precision, recall, thresholds = precision_recall_curve(T, S)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold_idx = f1_scores.argmax()
    best_threshold = thresholds[best_threshold_idx]
    Y_adjusted = (S >= best_threshold).astype(int)
    precision_adj, recall_adj, _ = precision_recall_curve(T, Y_adjusted)
    PRC = auc(recall_adj, precision_adj)
    return PRC

if __name__ == "__main__":
    import os

    data_select = "Da_to_Da"  # Da_to_Da
    iteration = 80
    decay_interval = 10
    batchs = 16
    lr = 2e-4  # 2e-4  # 5e-4
    weight_decay = 1e-4  # 1e-4  # 0.07
    lr_decay = 0.5
    setting = "Da_to_Da_dta3"
    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    dataset_train, dataset_test, p_LMs, d_LMs = data_load(data_select, device)
    setup_seed(2023)
    model = Representation_model(3, 512, 192, 64, batchs, device).to(device)
    target_model_state_dict = model.state_dict()
    source_model_state_dict = torch.load("output/model/pretrain_new_2")
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
            'CI\tMSE')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()
    ci1 = 0
    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train, p_LMs, d_LMs)
        # AUC, PRC, precision, recall = tester.test(dataset_test, p_LMs, d_LMs)
        CI, MSE = tester.test(dataset_test, p_LMs, d_LMs)

        end = timeit.default_timer()
        time = end - start

        CIs = [epoch, time, loss_train, CI, MSE]
        tester.save_AUCs(CIs, file_AUCs)
        print('\t'.join(map(str, CIs)))
        if ci1 < CI:
            ci1 = CI
            tester.save_model(model, file_model)
