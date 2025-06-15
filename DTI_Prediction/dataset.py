import numpy as np
from sklearn.utils import shuffle
import os

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset_strict(dataset, train_ratio=0.833):
    # 分成正负样本
    positives = [d for d in dataset if d[4] >= 12.1]
    negatives = [d for d in dataset if d[4] < 12.1]

    def split_samples(samples):
        train_set, test_set = [], []
        seen_proteins, seen_smiles = set(), set()

        for sample in samples:
            _, _, _, protein, _, _, smile = sample
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
    print(len(positives))
    print(len(negatives))
    pos_train, pos_test = split_samples(positives)
    neg_train, neg_test = split_samples(negatives)

    # 合并后打乱
    dataset_train = shuffle_dataset(pos_train + neg_train, seed=42)
    dataset_test = shuffle_dataset(pos_test + neg_test, seed=42)

    return dataset_train, dataset_test

def split(dir_input, input_file):
    sequences = np.load(dir_input + 'sequences.npy', allow_pickle=True)
    smiles = np.load(dir_input + 'smiles.npy', allow_pickle=True)
    molecule_words = np.load(dir_input + 'molecule_words.npy', allow_pickle=True)
    molecule_atoms = np.load(dir_input + 'molecule_atoms.npy', allow_pickle=True)
    molecule_adjs = np.load(dir_input + 'molecule_adjs.npy', allow_pickle=True)
    proteins = np.load(dir_input + 'proteins.npy', allow_pickle=True)
    interactions = np.load(dir_input + 'affinities.npy', allow_pickle=True)

    dataset = list(zip(molecule_words, molecule_atoms, molecule_adjs, proteins, interactions, sequences, smiles))
    dataset = shuffle_dataset(dataset, 42)

    dataset_train, dataset_test = split_dataset_strict(dataset, train_ratio=0.833)

    molecule_words_train, molecule_atoms_train, molecule_adjs_train = [], [], []
    proteins_train, interactions_train, sequences_train, smiles_train = [], [], [], []

    molecule_words_test, molecule_atoms_test, molecule_adjs_test = [], [], []
    proteins_test, interactions_test, sequences_test, smiles_test = [], [], [], []

    for item in dataset_train:
        mw, ma, madj, p, i, s, smi = item
        molecule_words_train.append(mw)
        molecule_atoms_train.append(ma)
        molecule_adjs_train.append(madj)
        proteins_train.append(p)
        interactions_train.append(i)
        sequences_train.append(s)
        smiles_train.append(smi)

    for item in dataset_test:
        mw, ma, madj, p, i, s, smi = item
        molecule_words_test.append(mw)
        molecule_atoms_test.append(ma)
        molecule_adjs_test.append(madj)
        proteins_test.append(p)
        interactions_test.append(i)
        sequences_test.append(s)
        smiles_test.append(smi)

    np.save(input_file + '/train/molecule_words', molecule_words_train)
    np.save(input_file + '/train/molecule_atoms', molecule_atoms_train)
    np.save(input_file + '/train/molecule_adjs', molecule_adjs_train)
    np.save(input_file + '/train/proteins', proteins_train)
    np.save(input_file + '/train/affinities', interactions_train)
    np.save(input_file + '/train/sequences', sequences_train)
    np.save(input_file + '/train/smiles', smiles_train)

    np.save(input_file + '/test/molecule_words', molecule_words_test)
    np.save(input_file + '/test/molecule_atoms', molecule_atoms_test)
    np.save(input_file + '/test/molecule_adjs', molecule_adjs_test)
    np.save(input_file + '/test/proteins', proteins_test)
    np.save(input_file + '/test/affinities', interactions_test)
    np.save(input_file + '/test/sequences', sequences_test)
    np.save(input_file + '/test/smiles', smiles_test)


# split("DrugBank/data_split/", "DrugBank")
# split("Human/data_split/", "Human")
# split("C.elegans/data_split/", "C.elegans")
# split("datasets/DAVIS/data_split_2/", "datasets/DAVIS")
split("datasets/KIBA/data_split_2/", "datasets/KIBA")
# split("Kiba/data_split/", "Kiba")

# 22154
# 94196

# 22603
# 95054

# 4337
# 21027