import numpy as np
import torch
import pickle

def load_tensor(file_name, dtype, device):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

def load_labels(file_name, dtype, device):
    labels = np.load(file_name + '.npy', allow_pickle=True)
    labels_new = []
    for label in labels:
        label = [int(char) for char in label]
        labels_new.append(label)
    return [dtype(d).to(device) for d in labels_new]

def train_data_load(dataset, name, device):
    prot_feature_train = load_tensor(dataset + '/train/' + name + '_Train_prot_features', torch.FloatTensor, device)
    label_train = load_labels(dataset + '/train/' + name + '_Train_labels', torch.LongTensor, device)
    return prot_feature_train, label_train

def test_data_load(dataset, name, device):
    prot_feature_test = load_tensor(dataset + '/test/' + name + '_Test_prot_features', torch.FloatTensor, device)
    label_test = load_labels(dataset + '/test/' + name + '_Test_labels', torch.LongTensor, device)
    return prot_feature_test, label_test

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def Source_ID(ID, length):
    source_label = torch.ones(length) * ID
    return source_label

def data_load(dataset_name, train_name, test_name, device):
    dataset = "datasets/" + dataset_name
    prot_feature_train, label_train = train_data_load(dataset, train_name, device)
    prot_feature_test, label_test = test_data_load(dataset, test_name, device)

    train_dataset = list(zip(prot_feature_train, label_train))
    train_dataset = shuffle_dataset(train_dataset, 1234)

    test_dataset = list(zip(prot_feature_test, label_test))
    test_dataset = shuffle_dataset(test_dataset, 1234)

    return train_dataset, test_dataset


# if __name__ == "__main__":
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#         print('The code uses GPU...')
#     else:
#         device = torch.device('cpu')
#         print('The code uses CPU!!!')
#     train_dataset, test_dataset = data_load(dataset_name="DNA", train_name="DNA-573", test_name="DNA-129", device=device)
#     print(len(train_dataset))
#     print(len(test_dataset))