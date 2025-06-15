import torch
import torch.nn as nn
from egnn import Protein3D_Representation
from MultiHeadAttention import MultiHeadAttention, Attention_ADJ
import torch.nn.functional as F
from Infoloss import InfoNCELoss
from spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified
import warnings
warnings.filterwarnings("ignore")
from evaluate import *
from KD import dkd_loss


class Decoder(nn.Module):
    def __init__(self, inc, dimension, outc, head, layers):
        super(Decoder, self).__init__()
        self.layers = layers
        self.act = nn.ReLU(inplace=True)
        self.linear_in = nn.Linear(inc, dimension, bias=False)
        self.LN_in = nn.LayerNorm(dimension)
        self.Attention = MultiHeadAttention(h=head, d_model=dimension)
        self.linear = nn.ModuleList([nn.Linear(dimension, dimension) for _ in range(layers)])
        self.LN_fea = nn.LayerNorm(dimension)
        self.linear_out = nn.Linear(dimension, outc)


    def forward(self, x):
        x = self.act(self.linear_in(x))
        x, atten = self.Attention(x, x, x)
        # x = self.LN_fea(x)
        for i in range(self.layers):
            x = self.act(self.linear[i](x))
        x = self.linear_out(x)#.view(B * N, -1)
        return x


class Representation_model(nn.Module):

    def __init__(self, num_layers, hidden_dim, decision_dim, edge_feat_dim, batch):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.represent = nn.Linear(1024, hidden_dim)
        self.ST_encoder = Protein3D_Representation(num_layers, hidden_dim, edge_feat_dim)
        self.atten = nn.ModuleList([Attention_ADJ(d_model=hidden_dim) for _ in range(3)])
        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        self.decoder = Decoder(inc=hidden_dim, dimension=decision_dim, outc=2, head=1, layers=2)
        # self.LM_decoder = Decoder(inc=hidden_dim, dimension=hidden_dim, outc=2, head=1, layers=2)
        self.batch = batch

    def forward(self, x_pos, h_LLM, prot_batchs, B, N):
        # print(x_pos.shape)
        h_vec = self.represent(h_LLM.view(B * N, -1))
        h_LM = self.represent(h_LLM)
        x_pos = x_pos.view(B * N, 3)
        prot_batchs = prot_batchs.view(B * N)
        h_fea, x_pos, all_h, edge_index = self.ST_encoder.forward(h_vec, x_pos, prot_batchs)
        h_fea = h_fea.view(B, N, self.hidden_dim)
        all_h2 = []
        attens = []
        for i in range(3):
            h_LMs, atten = self.atten[i](h_LM, h_LM, h_LM)
            attens.append(atten)
            h_LM = self.norms1[i](h_LM + h_LMs)
            all_h2.append(h_LM.view(B * N, self.hidden_dim))
        
        all_h = torch.cat(all_h, dim=0)#.squeeze(1)
        all_h2 = torch.cat(all_h2, dim=0).squeeze(1)
        h_st_out_cls = self.decoder(h_fea)
        h_lm_out_cls = self.decoder(h_LM)

        return h_st_out_cls.view(B * N, -1), h_lm_out_cls.view(B * N, -1), all_h, all_h2, attens

    def BSP_LM(self, h_LM, B, N): # language model predict BS
        # print(x_pos.shape)
        h_LM = self.represent(h_LM)
        for i in range(self.num_layers):
            h_LMs, atten = self.atten[i](h_LM, h_LM, h_LM)
            h_LM = self.norms1[i](h_LM + h_LMs)
        h_lm_out_cls = self.decoder(h_LM)
        return h_lm_out_cls.view(B * N, -1)

    def LM_generate(self, h_LM, B, N): # language model hide feature generation
        h_LM = self.represent(h_LM)
        attens = []
        for i in range(3):
            h_LMs, atten = self.atten[i](h_LM, h_LM, h_LM)
            attens.append(atten)
            h_LM = self.norms1[i](h_LM + h_LMs)
        return h_LM, attens

    def ST_generate(self, h_LM, x_pos, prot_batchs, B, N): # stuctural hide feature generation
        h_vec = self.represent(h_LM.view(B * N, -1))
        x_pos = x_pos.view(B * N, 3)
        prot_batchs = prot_batchs.view(B * N)
        h_fea, x_pos, all_h, edge_index = self.ST_encoder.forward(h_vec, x_pos, prot_batchs)
        h_fea = h_fea.view(B, N, self.hidden_dim)
        return h_fea, edge_index

    def __call__(self, data, device, train=True, LM_only=False):
        # print(np.array(data).shape)
        if LM_only:
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
        else:
            res_seqs, res_cooss, prot_features, prot_batchs, BS, B, N = data[0], data[1], data[2], data[3], data[4], data[5], data[6]
            B_data = B
            h_st_out_cls, h_lm_out_cls, all_h, all_h2, atten \
                = self.forward(res_cooss.to(device), prot_features.to(device), prot_batchs.to(device), B_data, N)
            SL1_loss = nn.MSELoss(reduction='mean')
            cls_loss = nn.CrossEntropyLoss()
            BStarget = torch.tensor(BS, dtype=torch.long).to(device).view(B * N)
            contrastive_loss = InfoNCELoss(temperature=1, normalize=True)
            res_cooss /= 100

            if train:
                loss1 = cls_loss(h_st_out_cls, BStarget)
                loss2 = cls_loss(h_lm_out_cls, BStarget) * 0.5 + dkd_loss(h_lm_out_cls, h_st_out_cls, BStarget) * 0.5
                loss3 = SL1_loss(all_h, all_h2)
                loss_all = loss1 + loss2 + loss3
                return loss_all, loss1, loss2, loss3
            else:
                return BS.view(B, N).cpu().detach().numpy(), h_st_out_cls.view(B, N, -1).cpu().detach().numpy(), h_lm_out_cls.view(B, N, -1).cpu().detach().numpy()
