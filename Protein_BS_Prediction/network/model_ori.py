import torch
import torch.nn as nn
from network.egnn import Protein3D_Representation
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
        self.ST_encoder = Protein3D_Representation(num_layers, hidden_dim, edge_feat_dim)
        self.atten = nn.ModuleList([Attention_ADJ(d_model=hidden_dim) for _ in range(num_layers)])
        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        """BS predictor"""
        self.decoder = Decoder2(inc=hidden_dim, dimension=decision_dim, outc=2, head=1, layers=2)
        # self.decoder2 = Decoder(inc=hidden_dim, dimension=decision_dim, outc=2, head=1, layers=2)
        self.batch = batch
        self.num_layers = num_layers

    def forward(self, x_pos, h_LLM, prot_batchs, B, N):
        # print(x_pos.shape)
        h_vec = self.represent(h_LLM.view(B * N, -1))
        h_LM = self.represent(h_LLM)
        x_pos = x_pos.view(B * N, 3)
        prot_batchs = prot_batchs.view(B * N)
        h_fea, x_pos, all_st, edge_index = self.ST_encoder.forward(h_vec, x_pos, prot_batchs)
        h_fea = h_fea.view(B, N, self.hidden_dim)

        all_lm = []
        attens = []
        for i in range(self.num_layers):
            h_LMs, atten = self.atten[i](h_LM, h_LM, None)
            attens.append(atten)
            h_LM = self.norms1[i](h_LM + h_LMs)
            all_lm.append(h_LM.view(B * N, self.hidden_dim))

        all_st = torch.cat(all_st, dim=0)  # .squeeze(1)
        all_lm = torch.cat(all_lm, dim=0).squeeze(1)
        h_st_out_cls = self.decoder(h_fea)
        h_lm_out_cls = self.decoder(h_LM)

        return h_st_out_cls.view(B * N, -1), h_lm_out_cls.view(B * N, -1), all_st.detach().clone(), all_lm, attens

    def BSP_LM(self, h_LM, B, N):  # language model predict BS
        # print(x_pos.shape)
        h_LM = self.represent(h_LM)
        for i in range(self.num_layers):
            h_LMs, atten = self.atten[i](h_LM, h_LM)
            h_LM = self.norms1[i](h_LM + h_LMs)
        h_lm_out_cls = self.decoder(h_LM)
        return h_lm_out_cls

    def ReSize(self, feature, N):
        molecule_ST = torch.zeros((N, 100, self.C), device=self.device)
        for i in range(N):
            C_L = feature[i].shape[0]
            if C_L >= 100:
                molecule_ST[i, :, :] = feature[i][0:100, :]
            else:
                molecule_ST[i, :C_L, :] = feature[i]
        return molecule_ST

    def LM_generate(self, h_LM, B, N):  # language model hide feature generation
        h_LM = self.represent(h_LM)
        attens = []
        for i in range(3):
            h_LMs, atten = self.atten[i](h_LM, h_LM, h_LM)
            attens.append(atten)
            h_LM = self.norms1[i](h_LM + h_LMs)
        return h_LM, attens

    def ST_generate(self, h_LM, x_pos, prot_batchs, B, N):  # stuctural hide feature generation
        h_vec = self.represent(h_LM.view(B * N, -1))
        x_pos = x_pos.view(B * N, 3)
        prot_batchs = prot_batchs.view(B * N)
        h_fea, x_pos, all_h, edge_index = self.ST_encoder.forward(h_vec, x_pos, prot_batchs)
        h_fea = h_fea.view(B, N, self.hidden_dim)
        return h_fea, edge_index

    def __call__(self, data, device, task, train=True):
        # print(np.array(data).shape)
        if task == "pretrain":
            res_seqs, res_cooss, prot_features, prot_batchs, BS, B, N = data[0], data[1], data[2], data[3], data[4], \
            data[5], data[6]
            B_data = B
            h_st_out_cls, h_lm_out_cls, all_st, all_lm, atten \
                = self.forward(res_cooss.to(device), prot_features.to(device), prot_batchs.to(device), B_data, N)
            SL1_loss = nn.MSELoss(reduction='mean')  # nn.SmoothL1Loss(reduction='mean') # nn.MSELoss(reduction='mean')
            cls_loss = nn.CrossEntropyLoss()
            BStarget = torch.tensor(BS, dtype=torch.long).to(device).view(B * N)

            if train:
                loss1 = cls_loss(h_st_out_cls, BStarget)
                loss2 = cls_loss(h_lm_out_cls, BStarget) * 0.5 + dkd_loss(h_lm_out_cls, h_st_out_cls, BStarget) * 0.5
                loss3 = SL1_loss(all_lm, all_st) * 2
                loss_all = loss1 + loss2 + loss3
                return loss_all, loss1, loss2, loss3
            else:
                return BS.view(B, N).cpu().detach().numpy(), h_st_out_cls.view(B, N,
                                                                               -1).cpu().detach().numpy(), h_lm_out_cls.view(
                    B, N, -1).cpu().detach().numpy()

        elif task == "BS prediction":
            prot_features, BS = data[0], data[1]
            B, N = prot_features.shape[0], prot_features.shape[1]
            h_lm_out_cls = self.BSP_LM(prot_features, B, N)
            cls_loss = nn.CrossEntropyLoss()
            BStarget = BS.view(B * N)

            if train:
                loss = cls_loss(h_lm_out_cls, BStarget)
                return loss
            else:
                return h_lm_out_cls, BStarget
