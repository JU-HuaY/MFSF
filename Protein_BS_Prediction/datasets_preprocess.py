from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
prot_tokenizer = T5Tokenizer.from_pretrained('prot_t5_xl_uniref50', do_lower_case=False)
prot_model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50").to(device)
prot_model.to(torch.float32)

def sequence_feature(sequences):
    protein_input = prot_tokenizer.batch_encode_plus([" ".join(sequences)], add_special_tokens=True, padding=True)
    p_IDS = torch.tensor(protein_input["input_ids"]).to(device)
    p_a_m = torch.tensor(protein_input["attention_mask"]).to(device)
    with torch.no_grad():
        prot_outputs = prot_model(input_ids=p_IDS, attention_mask=p_a_m)
    prot_feature = prot_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()
    return prot_feature

# protein-peptide
def toNum(l):
    l = [float(i) for i in l]
    return l
def load_tsv_format_data(filename, skip_head=True):
    sequences = []
    labels = []
    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            l = line.split('\t')
            sequences.append(l[0])
            labels.append(l[1])
    return sequences, labels

# protein-DNA/RNA/AB
def load_fasta_format_data(files):
    seq_ids, sequences, labels = [], [], []
    input_file = SeqIO.parse(files, 'fasta')
    for seq in input_file:
        seq_ids.append(seq.id)
        squence_labels = str(seq.seq)
        L = int(len(squence_labels)/2)
        squence = squence_labels[0: L]
        label = squence_labels[L: 2*L]
        sequences.append(squence)
        labels.append(label)
    return sequences, labels

def data_rewrite(path, subsets, filename, fasta=True):
    files = path + filename
    if fasta == True:
        sequences, labels = load_fasta_format_data(files + ".txt")
    else:
        sequences, labels = load_tsv_format_data(files + ".tsv")
    prot_features = []
    for i in range(len(sequences)):
        prot_features.append(sequence_feature(sequences[i]))
        print(i)
    np.save(path + subsets + filename + '_prot_features', prot_features)
    np.save(path + subsets + filename + '_labels', labels)

if __name__ == "__main__":
    path = "datasets/protein-peptide/"
    subsets = "test/" # train or test
    filename = "Dataset2_test"
    data_rewrite(path=path, subsets=subsets, filename=filename, fasta=False)
    print("done!")
