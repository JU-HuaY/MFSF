# -*- coding: utf-8 -*-
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

torch.multiprocessing.set_start_method('spawn')
import hashlib


def hash_string_to_int(s):
    hash_object = hashlib.sha256(s.encode('utf-8'))
    hash_digest = hash_object.hexdigest()
    hash_int = int(hash_digest, 16) % (2 ** 31)
    return hash_int


def pack(molecule_atoms, molecule_adjs, sequences, smiles, labels, p_LMs, d_LMs, device, sources=None):
    proteins_len = 1200
    drug_len = 100
    atoms_len = 0
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

    # molecule_atoms_new = torch.zeros((N, drug_len, 75), device=device)
    # i = 0
    # for atom in molecule_atoms:
    # a_len = atom.shape[0]
    # if a_len >= drug_len:
    # molecule_atoms_new[i, :, :] = atom[:drug_len, :]
    # else:
    # molecule_atoms_new[i, :a_len, :] = atom
    # i += 1

    # molecule_adjs_new = torch.zeros((N, drug_len, drug_len), device=device)
    # i = 0
    # for adj in molecule_adjs:
    # a_len = adj.shape[0]
    # adj = adj + torch.eye(a_len, device=device)
    # if a_len >= drug_len:
    # molecule_adjs_new[i, :, :] = adj[:drug_len, :drug_len]
    # else:
    # molecule_adjs_new[i, :a_len, :a_len] = adj
    # i += 1

    protein_LMs = []
    molecule_LMs = []
    for sequence in sequences:
        protein_LMs.append(p_LMs[sequence])

    for smile in smiles:
        molecule_LMs.append(d_LMs[smile])

    protein_LM = torch.zeros((N, proteins_len, 1024), device=device)
    molecule_LM = torch.zeros((N, drug_len, 768), device=device)
    # print(d_l)
    for i in range(N):
        C_L = molecule_LMs[i].shape[0]
        if C_L >= d_l:
            molecule_LM[i, :, :] = torch.tensor(molecule_LMs[i][0:d_l, :]).to(device)
        else:
            # molecule_LM[i, :C_L, :] = torch.tensor(molecule_LMs[i]).to(device)
            seed = hash_string_to_int(smiles[i])
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            noise_molecule = torch.randn((drug_len, 768), generator=gen, device=device) * 1e-6
            noise_molecule[:C_L, :] = torch.tensor(molecule_LMs[i]).to(device)
            molecule_LM[i, :, :] = noise_molecule

        P_L = protein_LMs[i].shape[0]
        if P_L >= proteins_len:
            protein_LM[i, :, :] = torch.tensor(protein_LMs[i][0:proteins_len, :]).to(device)
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


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}

        for name, param in model.named_parameters():
            self.shadow[name] = param.clone()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param

    def apply_shadow(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(self.shadow[name])

    def restore(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(self.shadow[name])


class Trainer(object):
    def __init__(self, model, batch_size, lr, weight_decay, ema):
        self.model = model
        self.ema = ema
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                     weight_decay=weight_decay)
        # self.schedule = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)
        self.batch_size = batch_size

    def train(self, dataset, p_LMs, d_LMs, epoch):
        import torch.nn.functional as F
        from scipy.stats import median_abs_deviation
        np.random.shuffle(dataset)
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
                    data_batch = (molecule_atoms, molecule_adjs, protein_LM, molecule_LM, labels, sources)
                    
                    
                    if epoch >= 50:
                        self.model.eval()
                        with torch.no_grad():
                            all_losses = []
                            all_labels = []
                            for j in range(molecule_atoms.shape[0]):
                                sample = tuple(x[j:j + 1] for x in data_batch)
                                sample_loss = self.model(sample, device, "DTI prediction")
                                all_losses.append(sample_loss.item())
                                all_labels.append(labels[j].item())
                        self.model.train()

                        losses_np = np.array(all_losses)
                        labels_np = np.array(all_labels)

                        keep_indices = [j for j, l in enumerate(labels_np) if l == 1]

                        zero_indices = [j for j, l in enumerate(labels_np) if l == 0]
                        if len(zero_indices) > 0:
                            zero_losses = losses_np[zero_indices]
                            median = np.median(zero_losses)
                            mad = median_abs_deviation(zero_losses) + 1e-6
                            threshold = median + 1.0 * mad
                            keep_zero_indices = [zero_indices[j] for j, l in enumerate(zero_losses) if l <= threshold]
                            keep_indices.extend(keep_zero_indices)

                        if len(keep_indices) < 2:
                            keep_indices = list(range(len(all_losses)))

                        data_batch = tuple(x[keep_indices] for x in data_batch)
                        
                        
                    loss = self.model(data_batch, device, "DTI prediction")  # .mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                    molecule_atoms, molecule_adjs, sequences, smiles, labels, sources = [], [], [], [], [], []
                else:
                    molecule_atoms, molecule_adjs, sequences, smiles, labels, sources = [], [], [], [], [], []
            else:
                continue

            if i % self.batch_size == 0 or i == N:
                self.optimizer.step()
                # self.schedule.step()
                self.optimizer.zero_grad()
                # self.ema.update()
            loss_total += loss.item()
        return loss_total



class Tester(object):
    def __init__(self, model, batch_size, ema=None):
        self.model = model
        self.batch_size = batch_size
        self.ema = ema

    def test(self, dataset, p_LMs, d_LMs, epoch):
        #if self.ema is not None and epoch > 12:
            #self.ema.apply_shadow()
        N = len(dataset)
        T, S, Y, S2, Y2, S3, Y3 = [], [], [], [], [], [], []
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
                # print(words.shape)
                data = (molecule_atoms, molecule_adjs, protein_LM, molecule_LM, labels, _)
                # print(self.model(data, train=False))
                correct_labels, ys = self.model(data, device, "DTI prediction", train=False)
                correct_labels = correct_labels.to('cpu').data.numpy()
                ys = ys.to('cpu').data.numpy()
                predicted_labels = list(map(lambda x: np.argmax(x), ys))
                # threshold = 0.40
                # predicted_labels = list(map(lambda x: 1 if x[1] >= threshold else 0, ys))
                predicted_scores = list(map(lambda x: x[1], ys))

                for j in range(len(correct_labels)):
                    T.append(correct_labels[j])
                    Y.append(predicted_labels[j])
                    S.append(predicted_scores[j])

                molecule_atoms, molecule_adjs, sequences, smiles, labels = [], [], [], [], []
            else:
                continue

        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        pre, rec, thresholds = precision_recall_curve(T, S)
        pr_auc = auc(rec, pre)
        return AUC, pr_auc, precision, recall

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


if __name__ == "__main__":
    import os
    from sklearn.model_selection import train_test_split

    data_select = "D_to_D" #Da_to_Da  K_to_K
    iteration = 120
    decay_interval = 10
    batchs = 16
    lr = 2e-4
    weight_decay = 1e-4  # 0.07 # 1e-4 #
    lr_decay = 0.5
    layer_gnn = 3
    drop = 0.05
    setting = "D_to_Dnew_2"#"Da_to_Da_6new_poly3" #_test1
    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    train_set, val_set, dataset_test, p_LMs, d_LMs = data_load(data_select, device)
    setup_seed(42)
    model = Representation_model(3, 512, 192, 64, batchs, device).to(device)
    target_model_state_dict = model.state_dict()
    source_model_state_dict = torch.load(
        "/mnt/fast/nobackup/scratch4weeks/yh01358/DTI_Prediction/output/model/pretrain_new_2")
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
    ema = EMA(model, decay=0.999)
    # model.parameters()
    # model = torch.nn.DataParallel(model, device_ids=[1], output_device=1)
    # model = model.module.to(torch.device('cpu'))
    trainer = Trainer(model, batchs, lr, weight_decay, ema)
    tester = Tester(model, batchs, ema)

    """Output files."""
    file_AUCs = '/mnt/fast/nobackup/scratch4weeks/yh01358/DTI_Prediction/output/result/AUCs--' + setting + '.txt'
    file_model = '/mnt/fast/nobackup/scratch4weeks/yh01358/DTI_Prediction/output/model/' + setting
    AUCs = ('Epoch\tTime(sec)\tLoss_train\t'
            'AUC\tPRC\tprecision\trecall')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()
    auc1 = 0
    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(train_set, p_LMs, d_LMs, epoch)
        AUC, PRC, precision, recall = tester.test(dataset_test, p_LMs, d_LMs, epoch)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train,
                AUC, PRC, precision, recall]
        tester.save_AUCs(AUCs, file_AUCs)
        # tester.save_model(model, file_model)
        print('\t'.join(map(str, AUCs)))
        if auc1 < AUC:
            auc1 = AUC
            tester.save_model(file_model)
