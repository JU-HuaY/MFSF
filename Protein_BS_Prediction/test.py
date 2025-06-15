import pickle
import sys
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
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.metrics import matthews_corrcoef

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

if __name__ == "__main__":
    batchs = 1
    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    model = Representation_model(3, 512, 256, 64, 128, batchs, device).to(device)
    target_model_state_dict = model.state_dict()
    model_state_dict = torch.load("output/model/RNA_RNA-545_RNA-161_2") # RNA_RNA-545_RNA-161_1 DNA_DNA-573_DNA-129_1
    model.load_state_dict(model_state_dict)
    DNA_sequence = "AAPALKEIFNVERLQHIASEMTAVYPAFDAKGFLKHAKAGLAELSVMQRMARVSESLHAVIPLDYPQTLTLLYALAPRLNSGFVSLFLPHYVASYGRDDFKRSMAALKYFTTFGSAEFAIRHFLLHDFQRTLAVMQAWSQDDNEHVRRLASEGSRPRLPWSFRLAEVQADPELCASILDHLKADSSLYVRKSVANHLNDITKDHPEWVLSLIEGWNLENPHTAWIARHALRSLIKQGNTRALTLMGAGAKAEVKIHHLMVTPAVINLGERINLSFTLESTAPAPQKLVVDYAIDYVKSTGHGAAKVFKLKAFSLGAGAQQHIRREQHIRDMTTRKHYPGRHVVHVLVNGERLGSAEFELRA"
    DNA_label_ = "0001110000000000000000000000000000000000000011100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000100100001111000000000000000000000000100000010010001000000000000000000000000010010000000000000000000000000000000000000000000000000000000000000000011000000101100000000000000000000001111001000000000000000000000000"
    RNA_sequence = "MHHHHHHSSGLVPRGSGMKETAAAKFERQHMDSPDLGTDDDDKAMADIGSENLYFQMWLTKLVLNPASRAARRDLANPYEMHRTLSKAVSRALEEGRERLLWRLEPARGLEPPVVLVQTLTEPDWSVLDEGYAQVFPPKPFHPALKPGQRLRFRLRANPAKRLAATGKRVALKTPAEKVAWLERRLEEGGFRLLEGERGPWVQILQDTFLEVRRKKDGEEAGKLLQVQAVLFEGRLEVVDPERALATLRRGVGPGKALGLGLLSVAP"
    RNA_label_ = "000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000001000000000000000000000000000000000000000000000000000000000010011010001101000000000000110000000000000000000000010000100000000110101000000000000000000000000000100000000000"
    sequence = RNA_sequence
    label_ = RNA_label_
    prot_feature = torch.FloatTensor(sequence_feature(sequence)).to(device)
    label = torch.tensor(list(map(int, label_)), dtype=torch.int64)
    prot_features, labels = prot_feature[:-1].unsqueeze(0).to(device), label.unsqueeze(0).to(device)
    data = (prot_features, labels)
    pred_bs, label = model(data, device, task="BS prediction", train=False)
    pred_bs_int = torch.argmax(pred_bs, dim=-1)
    mcc_stu = matthews_corrcoef(label.detach().cpu().numpy(), pred_bs_int.detach().cpu().numpy())
    print(mcc_stu)
    print(''.join(map(str, pred_bs_int.tolist())))
    print(''.join(map(str, label.tolist())))
    print(torch.nonzero(pred_bs_int == 1).squeeze())
    print(torch.nonzero(label == 1).squeeze())


