import numpy as np
import torch
import pickle
from sklearn.model_selection import train_test_split


def load_tensor(file_name, dtype, device):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

def train_data_load(dataset, device, DTI=True):
    molecule_atoms_train = load_tensor(dataset + '/train/molecule_atoms', torch.LongTensor, device)
    molecule_adjs_train = load_tensor(dataset + '/train/molecule_adjs', torch.LongTensor, device)
    sequence_train = np.load(dataset + '/train/sequences.npy')
    smiles_train = np.load(dataset + '/train/smiles.npy')
    if DTI == True:
        interactions_train = load_tensor(dataset + '/train/interactions', torch.LongTensor, device)
    else:
        interactions_train = load_tensor(dataset + '/train/affinities', torch.FloatTensor, device)

    with open(dataset + '/p_LM.pkl', 'rb') as p:
        p_LM = pickle.load(p)

    with open(dataset + '/d_LM.pkl', 'rb') as d:
        d_LM = pickle.load(d)

    return molecule_atoms_train, molecule_adjs_train, sequence_train, smiles_train, p_LM, d_LM, interactions_train

def test_data_load(dataset, device, DTI=True):
    molecule_atoms_test = load_tensor(dataset + '/test/molecule_atoms', torch.LongTensor, device)
    molecule_adjs_test = load_tensor(dataset + '/test/molecule_adjs', torch.LongTensor, device)
    sequence_test = np.load(dataset + '/test/sequences.npy')
    smiles_test = np.load(dataset + '/test/smiles.npy')
    if DTI == True:
        interactions_test = load_tensor(dataset + '/test/interactions', torch.LongTensor, device)
    else:
        interactions_test = load_tensor(dataset + '/test/affinities', torch.FloatTensor, device)

    with open(dataset + '/p_LM.pkl', 'rb') as p:
        p_LM = pickle.load(p)
    with open(dataset + '/d_LM.pkl', 'rb') as d:
        d_LM = pickle.load(d)

    return molecule_atoms_test, molecule_adjs_test, sequence_test, smiles_test, p_LM, d_LM, interactions_test

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def Source_ID(ID, length):
    source_label = torch.ones(length) * ID
    return source_label
    

def split_dataset_strict(dataset, train_ratio=0.8, seed=42):
    positives = [d for d in dataset if d[4] > 5.0]
    negatives = [d for d in dataset if d[4] == 5.0]

    def split_samples(samples):
        samples = shuffle_dataset(samples, seed=seed)
        train_set, test_set = [], []
        seen_proteins, seen_smiles = set(), set()

        for sample in samples:
            _, _, protein, smile, _, _ = sample
            p_key = protein.tobytes()
            s_key = smile.tobytes()

            if len(train_set) < int(len(samples) * train_ratio):
                train_set.append(sample)
                seen_proteins.add(p_key)
                seen_smiles.add(s_key)
            else:
                if p_key in seen_proteins and s_key in seen_smiles:
                    test_set.append(sample)
                else:
                    train_set.append(sample)
                    seen_proteins.add(p_key)
                    seen_smiles.add(s_key)

        return train_set, test_set

    pos_train, pos_test = split_samples(positives)
    neg_train, neg_test = split_samples(negatives)

    dataset_train = shuffle_dataset(pos_train + neg_train, seed=seed)
    dataset_test = shuffle_dataset(pos_test + neg_test, seed=seed)

    return dataset_train, dataset_test    


def merge2(train_list, test_list, device, DTI=True):
    molecule_atoms_trains, molecule_adjs_trains, proteins_trains, sequence_trains, smiles_trains, interactions_trains = [], [], [], [], [], []
    molecule_atoms_tests, molecule_adjs_tests, proteins_tests, sequence_tests, smiles_tests, interactions_tests = [], [], [], [], [], []
    p_LMs, d_LMs = {}, {}
    train_source = []
    ID = 0
    for dataset in train_list:
        molecule_atoms_train, molecule_adjs_train, sequence_train, smiles_train, p_LM, d_LM, interactions_train = train_data_load(dataset, device, DTI)
        molecule_atoms_trains.extend(molecule_atoms_train)
        molecule_adjs_trains.extend(molecule_adjs_train)
        sequence_trains.extend(sequence_train)
        smiles_trains.extend(smiles_train)
        p_LMs.update(p_LM)
        d_LMs.update(d_LM)
        interactions_trains.extend(interactions_train)
        length = len(molecule_atoms_train)
        train_source.extend(Source_ID(ID, length))
        ID += 1

    for dataset in test_list:
        molecule_atoms_test, molecule_adjs_test, sequence_test, smiles_test, p_LM, d_LM, interactions_test = test_data_load(dataset, device, DTI)
        molecule_atoms_tests.extend(molecule_atoms_test)
        molecule_adjs_tests.extend(molecule_adjs_test)
        sequence_tests.extend(sequence_test)
        smiles_tests.extend(smiles_test)
        p_LMs.update(p_LM)
        d_LMs.update(d_LM)
        interactions_tests.extend(interactions_test)

    train_dataset = list(zip(molecule_atoms_trains, molecule_adjs_trains, sequence_trains, smiles_trains, interactions_trains, train_source))
    # train_dataset = shuffle_dataset(train_dataset, 1234)
    dataset_train_pos, dataset_train_neg = split_dataset(train_dataset, 0.5)
    train_set_pos, val_set_pos = train_test_split(dataset_train_pos, test_size=0.2, random_state=42)
    train_set_neg, val_set_neg = train_test_split(dataset_train_neg, test_size=0.2, random_state=42)
    train_set = shuffle_dataset(train_set_pos+train_set_neg, 42)
    val_set = shuffle_dataset(val_set_pos+val_set_neg, 42)

    test_dataset = list(zip(molecule_atoms_tests, molecule_adjs_tests, sequence_tests, smiles_tests, interactions_tests))
    test_dataset = shuffle_dataset(test_dataset, 42)

    return train_set, val_set, test_dataset, p_LMs, d_LMs

def merge(train_list, test_list, device, DTI=True):
    molecule_atoms_trains, molecule_adjs_trains, proteins_trains, sequence_trains, smiles_trains, interactions_trains = [], [], [], [], [], []
    molecule_atoms_tests, molecule_adjs_tests, proteins_tests, sequence_tests, smiles_tests, interactions_tests = [], [], [], [], [], []
    p_LMs, d_LMs = {}, {}
    train_source = []
    ID = 0
    for dataset in train_list:
        molecule_atoms_train, molecule_adjs_train, sequence_train, smiles_train, p_LM, d_LM, interactions_train = train_data_load(dataset, device, DTI)
        molecule_atoms_trains.extend(molecule_atoms_train)
        molecule_adjs_trains.extend(molecule_adjs_train)
        sequence_trains.extend(sequence_train)
        smiles_trains.extend(smiles_train)
        p_LMs.update(p_LM)
        d_LMs.update(d_LM)
        interactions_trains.extend(interactions_train)
        length = len(molecule_atoms_train)
        train_source.extend(Source_ID(ID, length))
        ID += 1

    for dataset in test_list:
        molecule_atoms_test, molecule_adjs_test, sequence_test, smiles_test, p_LM, d_LM, interactions_test = test_data_load(dataset, device, DTI)
        molecule_atoms_tests.extend(molecule_atoms_test)
        molecule_adjs_tests.extend(molecule_adjs_test)
        sequence_tests.extend(sequence_test)
        smiles_tests.extend(smiles_test)
        p_LMs.update(p_LM)
        d_LMs.update(d_LM)
        interactions_tests.extend(interactions_test)

    train_dataset = list(zip(molecule_atoms_trains, molecule_adjs_trains, sequence_trains, smiles_trains, interactions_trains, train_source))
    # train_dataset = shuffle_dataset(train_dataset, 42)
    train_set, val_set = train_test_split(train_dataset, test_size=0.2, random_state=42) #split_dataset_strict(train_dataset, train_ratio=0.8, seed=42) # train_test_split(train_dataset, test_size=0.2, random_state=42)

    test_dataset = list(zip(molecule_atoms_tests, molecule_adjs_tests, sequence_tests, smiles_tests, interactions_tests))
    test_dataset = shuffle_dataset(test_dataset, 42)

    return train_set, val_set, test_dataset, p_LMs, d_LMs


def data_load(data_select, device):
    # p_LMs, d_LMs = {}, {}
    if data_select == "D_to_D":
        train_list = ["/mnt/fast/nobackup/scratch4weeks/yh01358/DTI_Prediction/datasets/Drugbank"]
        test_list = ["/mnt/fast/nobackup/scratch4weeks/yh01358/DTI_Prediction/datasets/Drugbank"]
        train_set, val_set, test_dataset, p_LMs, d_LMs = merge2(train_list, test_list, device)
    elif data_select == "Da_to_Da":
        train_list = ["/mnt/fast/nobackup/scratch4weeks/yh01358/DTI_Prediction/datasets/Davis"]#, "datasets/Davis/split2"]
        test_list = ["/mnt/fast/nobackup/scratch4weeks/yh01358/DTI_Prediction/datasets/Davis"]
        train_set, val_set, test_dataset, p_LMs, d_LMs = merge(train_list, test_list, device, DTI=False)
    else:
        train_list = ["/mnt/fast/nobackup/scratch4weeks/yh01358/DTI_Prediction/datasets/KIBA"]#, "datasets/KIBA/split2"]
        test_list = ["/mnt/fast/nobackup/scratch4weeks/yh01358/DTI_Prediction/datasets/KIBA"]
        train_set, val_set, test_dataset, p_LMs, d_LMs = merge(train_list, test_list, device, DTI=False)

    return train_set, val_set, test_dataset, p_LMs, d_LMs

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
    train_set, val_set, test_dataset, p_LMs, d_LMs = data_load("Da_to_Da", device)
    i = 0
    all = 0
    for data in train_dataset:
        all+=1
        molecule_atom, molecule_adj, sequence, smile, label, _ = data
        if label == 5:
            print(label)
            i+=1
    print(i)
    print(all-i)
