import torch
import torch.nn as nn
from network.MultiHeadAttention import MultiHeadAttention, Attention_ADJ
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")
from network.KD import dkd_loss
from network.utils import *


class Representation_model(nn.Module):

    def __init__(self, num_layers, hidden_dim, uni_dim1, uni_dim2, batch, device):
        super().__init__()
        """pretrain"""
        self.represent = nn.Linear(1024, hidden_dim)
        self.atten = nn.ModuleList([Attention_ADJ(d_model=hidden_dim, K=80) for _ in range(3)])
        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        self.decoder = Decoder(inc=hidden_dim, dimension=256, outc=2, head=1, layers=2)
        for p in self.parameters():
            p.requires_grad = False
        self.att = nn.Softmax(dim=-1)
        """DTI(A) predictor"""
        self.num_layers = num_layers
        self.uniform_protein = nn.Linear(hidden_dim, uni_dim2) #Encoder(hidden_dim, uni_dim2, 64, 3, groups=32, pad1=9, pad2=5) #nn.Linear(hidden_dim, uni_dim2)
        self.encoder_protein = Encoder(1024, uni_dim1, 128, 3, groups=64, pad1=9, pad2=5) # 9 5
        self.encoder_drug = Encoder(768, uni_dim1, 64, 3, groups=32, pad1=7, pad2=3)
        self.atom_rep = Atom_rep(uni_dim2, device)
        self.GAT = GAT(uni_dim2, uni_dim2, 3).to("cuda")
        self.gnn_act = nn.GELU()
        self.C = uni_dim2
        self.cross_attention = cross_attention(uni_dim1, uni_dim2)
        self.se_p = SEBlock(channel=256)
        # self.se_d = SEBlock(channel=256)
        self.DTI_predictor = Mix_classfier(classes=2, feature_length=uni_dim1 + uni_dim2)
        self.DTA_predictor = Mix_classfier_dta(classes=1, feature_length=uni_dim1 + uni_dim2, layers=3, DTA=True)
        self.device = device

    def ReSize(self, feature, N):
        molecule_ST = torch.zeros((N, 100, self.C), device=self.device)
        for i in range(N):
            C_L = feature[i].shape[0]
            if C_L >= 100:
                molecule_ST[i, :, :] = feature[i][0:100, :]
            else:
                molecule_ST[i, :C_L, :] = feature[i]
        return molecule_ST

    def forward(self, h_LM, molecule_LM, molecule_atoms, molecule_adjs):
        """Protein feature extractor"""
        protein_LM = self.encoder_protein(h_LM.permute(0, 2, 1)).permute(0, 2, 1)
        h_LM_ST = self.represent(h_LM)
        for i in range(self.num_layers):
            h_LM_STs, atten = self.atten[i](h_LM_ST, h_LM_ST, None)
            h_LM_ST = self.norms1[i](h_LM_ST + h_LM_STs)
        protein_bs = self.att(self.decoder(h_LM_ST))[:, :, 1]
        h_LM_ST = self.uniform_protein(h_LM_ST)
        proteins = torch.cat((protein_LM, h_LM_ST), 2)
        proteins = self.se_p(proteins.permute(0, 2, 1)).permute(0, 2, 1)
        """Drug feature extractor"""
        molecule_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        N = molecule_atoms.shape[0]
        molecule_vec = self.atom_rep(molecule_atoms, N)
        molecule_ST = self.GAT(molecule_vec, molecule_adjs)
        molecule_ST = self.ReSize(molecule_ST, N)
        molecule_ST = self.cross_attention(molecule_ST, molecule_LM)
        molecules = torch.cat((molecule_LM, molecule_ST), 2)
        # molecules = self.se_d(molecules.permute(0, 2, 1)).permute(0, 2, 1)
        '''DECISION'''
        DTI = self.DTI_predictor(proteins, molecules, protein_bs)
        return DTI

    def DTA_prediction(self, h_LM, molecule_LM, molecule_atoms, molecule_adjs):
        """Protein feature extractor"""
        protein_LM = self.encoder_protein(h_LM.permute(0, 2, 1)).permute(0, 2, 1)
        h_LM_ST = self.represent(h_LM)
        for i in range(self.num_layers):
            h_LM_STs, atten = self.atten[i](h_LM_ST, h_LM_ST, None)
            h_LM_ST = self.norms1[i](h_LM_ST + h_LM_STs)
        protein_bs = self.att(self.decoder(h_LM_ST))[:, :, 1]
        h_LM_ST = self.uniform_protein(h_LM_ST)
        proteins = torch.cat((protein_LM, h_LM_ST), 2)
        proteins = self.se_p(proteins.permute(0, 2, 1)).permute(0, 2, 1)
        """Drug feature extractor"""
        molecule_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        N = molecule_atoms.shape[0]
        molecule_vec = self.atom_rep(molecule_atoms, N)
        molecule_ST = self.GAT(molecule_vec, molecule_adjs)
        # molecule_ST = self.ReSize(molecule_ST, N)
        molecule_ST = self.cross_attention(molecule_ST, molecule_LM)
        molecules = torch.cat((molecule_LM, molecule_ST), 2)
        # molecules = self.se_d(molecules.permute(0, 2, 1)).permute(0, 2, 1)
        '''DECISION'''
        DTA = self.DTA_predictor(proteins, molecules, protein_bs)
        return DTA

    def LM_generate(self, h_LM, B, N):  # language model hide feature generation
        h_LM = self.represent(h_LM)
        attens = []
        for i in range(3):
            h_LMs, atten = self.atten[i](h_LM, h_LM, h_LM)
            attens.append(atten)
            h_LM = self.norms1[i](h_LM + h_LMs)
        return h_LM, attens

    def __call__(self, data, device, task, train=True):
        # print(np.array(data).shape)
        if task == "DTI prediction":
            molecule_atoms, molecule_adjs, protein_LM, molecule_LM, labels, _ = data[0], data[1], data[2], data[3], \
            data[4], data[5]
            DTI_pre = self.forward(protein_LM, molecule_LM, molecule_atoms, molecule_adjs)
            # labels = a2i(labels, datasets="Davis") # davis and kiba
            labels = torch.LongTensor(labels.to('cpu').numpy()).cuda() # drugbank
            # criterion = FocalLoss(alpha=0.6, gamma=2.0, reduction='mean') # nn.CrossEntropyLoss(weight=torch.FloatTensor([0.3, 0.7]).cuda()) #FocalLoss(alpha=0.7, gamma=2.0, reduction='mean')  # davis and kiba 
            criterion = PolyLoss(epsilon=1.0) # nn.CrossEntropyLoss(label_smoothing=0.1) #AsymmetricLossOptimized(gamma_neg=2.5, gamma_pos=1, clip=0.05)  #  # label_smoothing=0.05
            if train:
                loss = criterion(DTI_pre, labels)  # davis and kiba , label_smoothing=0.05
                # loss = F.cross_entropy(DTI_pre, labels)
                return loss
            else:
                ys = F.softmax(DTI_pre, 1)
                return labels, ys

        elif task == "DTA_prediction":
            molecule_atoms, molecule_adjs, protein_LM, molecule_LM, labels, _ = data[0], data[1], data[2], data[3], \
            data[4], data[5]
            DTA_pre = self.DTA_prediction(protein_LM, molecule_LM, molecule_atoms, molecule_adjs)
            labels = labels.float().cuda()
            SML_loss = nn.SmoothL1Loss(
                reduction='mean')  # nn.SmoothL1Loss(reduction='mean') # nn.MSELoss(reduction='mean') #
            if train:
                loss = SML_loss(DTA_pre.squeeze(), labels)
                return loss
            else:
                return labels, DTA_pre.squeeze()


def a2i(affinities, datasets="Davis"):
    N = len(affinities)
    interactions = torch.zeros(N, device='cuda')
    if datasets == "KIBA":
        for i in range(N):
            if affinities[i] > 12.1:
                interactions[i] = 1

    else:
        for i in range(N):
            if affinities[i] >= 6.0:
                interactions[i] = 1
    return torch.LongTensor(interactions.to('cpu').numpy()).cuda()
