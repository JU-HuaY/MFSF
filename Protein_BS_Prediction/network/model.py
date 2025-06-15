import torch
import torch.nn as nn
from network.MultiHeadAttention import MultiHeadAttention, Attention_ADJ
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
from network.KD import dkd_loss
from network.utils import *


class Representation_model(nn.Module):

    def __init__(self, num_layers, hidden_dim, decision_dim, edge_feat_dim, uni_dim, batch, device):
        super().__init__()
        """pretrain"""
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.represent = nn.Linear(1024, hidden_dim)
        self.atten = nn.ModuleList([Attention_ADJ(d_model=hidden_dim) for _ in range(3)])
        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        # for p in self.parameters():
        #     p.requires_grad = False
        """BS predictor"""
        self.encoder_protein = Encoder(1024, hidden_dim, 128, 3, groups=64, pad1=9, pad2=5)
        self.decoder = Decoder(inc=hidden_dim, dimension=decision_dim, outc=2, head=1, layers=2)
        # self.LM_decoder = Decoder(inc=hidden_dim, dimension=hidden_dim, outc=2, head=1, layers=2)
        self.batch = batch

    def forward(self, h_LM, B, N): # language model predict BS
        # print(x_pos.shape)
        protein_LM = self.encoder_protein(h_LM.permute(0, 2, 1)).permute(0, 2, 1)
        h_LM = self.represent(h_LM)
        for i in range(self.num_layers):
            h_LMs, atten = self.atten[i](h_LM, h_LM)
            h_LM = self.norms1[i](h_LM + h_LMs)
        h_lm_out_cls = self.decoder(h_LM, protein_LM)
        return h_lm_out_cls.view(B * N, -1)

    def ReSize(self, feature, N):
        molecule_ST = torch.zeros((N, 100, self.C), device=self.device)
        for i in range(N):
            C_L = feature[i].shape[0]
            if C_L >= 100:
                molecule_ST[i, :, :] = feature[i][0:100, :]
            else:
                molecule_ST[i, :C_L, :] = feature[i]
        return molecule_ST

    def LM_generate(self, h_LM, B, N): # language model hide feature generation
        h_LM = self.represent(h_LM)
        attens = []
        for i in range(3):
            h_LMs, atten = self.atten[i](h_LM, h_LM, h_LM)
            attens.append(atten)
            h_LM = self.norms1[i](h_LM + h_LMs)
        return h_LM, attens


    def __call__(self, data, device, task, train=True):
        # print(np.array(data).shape)
        prot_features, BS = data[0], data[1]
        B, N = prot_features.shape[0], prot_features.shape[1]
        h_lm_out_cls = self.forward(prot_features, B, N)
        cls_loss = nn.CrossEntropyLoss()
        BStarget = BS.view(B * N)

        if train:
            loss = cls_loss(h_lm_out_cls, BStarget)
            return loss
        else:
            return h_lm_out_cls, BStarget