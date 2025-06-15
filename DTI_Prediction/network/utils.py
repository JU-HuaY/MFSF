import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from network.MultiHeadAttention import MultiHeadAttention, Attention_ADJ
import numpy as np
from network.spd import SPDTangentSpace, SPDRectified


class EncoderLayer(nn.Module):
    def __init__(self, i_channel, o_channel, growth_rate, groups, pad2=7):
        super(EncoderLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=i_channel, out_channels=o_channel, kernel_size=(2 * pad2 + 1), stride=1,
                               groups=groups, padding=pad2,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(i_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=o_channel, out_channels=growth_rate, kernel_size=(2 * pad2 + 1), stride=1,
                               groups=groups, padding=pad2,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(o_channel)

    def forward(self, x):
        # xn = self.bn1(x)
        xn = self.relu(x)
        xn = self.conv1(xn)
        xn = self.bn2(xn)
        xn = self.relu(xn)
        xn = self.conv2(xn)
        return torch.cat([x, xn], 1)


class Encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=15, pad2=7):
        super(Encoder, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv1d(in_channels=inc, out_channels=inc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1,
                                 bias=False)
        self.dense_cnn = nn.ModuleList(
            [EncoderLayer(inc + growth_rate * i_la, inc + (growth_rate // 2) * i_la, growth_rate, groups, pad2) for i_la
             in range(layers)])
        self.conv_out = nn.Conv1d(in_channels=inc + growth_rate * layers, out_channels=outc, kernel_size=(pad1 * 2 + 1),
                                  stride=1, padding=pad1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.layers):
            x = self.dense_cnn[i](x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        return x


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.QK = nn.Linear(in_features, out_features).to("cuda")
        self.V = nn.Linear(in_features, out_features).to("cuda")
        self.drop = nn.Dropout(0.1)
        self.act = nn.GELU()

    def forward(self, inp, adj):
        h_qk = self.QK(inp)
        h_v = self.act(self.V(inp))
        a_input = torch.matmul(h_qk, h_qk.permute(0, 2, 1))
        scale = h_qk.size(-1) ** -0.5
        attention_adj = torch.sigmoid(a_input * scale) * adj
        h_prime = torch.matmul(attention_adj, h_v)
        return h_prime


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, gnn_layer):
        super(GAT, self).__init__()
        self.attentions = [GATLayer(n_feat, n_hid) for _ in
                           range(gnn_layer)]
        self.gnn_layer = gnn_layer
        # self.layernorm = nn.LayerNorm(n_feat) #D

    def forward(self, x, adj):
        for i in range(self.gnn_layer):
            x = self.attentions[i](x, adj) + x
            # x = self.layernorm(x) #D
        return x


class Atom_rep(nn.Module):
    def __init__(self, channels, device, atom_classes=16, atom_hidden=33):
        super(Atom_rep, self).__init__()
        self.embed_comg = nn.Embedding(atom_classes, atom_hidden)
        self.device = device
        self.channel = channels

    def forward(self, molecule_atoms, N):
        molecule_vec = torch.zeros((molecule_atoms.shape[0], molecule_atoms.shape[1], self.channel), device=self.device)
        for i in range(N):
            fea = torch.zeros((molecule_atoms.shape[1], self.channel), device=self.device)
            atom_fea = molecule_atoms[i][:, 0:16]
            p = torch.argmax(atom_fea, dim=1)
            com = self.embed_comg(p)
            oth1 = molecule_atoms[i][:, 44:75]
            tf = F.normalize(oth1, dim=1)
            fea[:, 0:33] = com
            fea[:, 33:64] = tf
            molecule_vec[i, :, :] = fea
        return molecule_vec


class cross_attention(nn.Module):
    def __init__(self, hidden1, hidden2):
        super(cross_attention, self).__init__()
        self.W_q = nn.Linear(hidden1, hidden2)
        self.W_k = nn.Linear(hidden2, hidden2)
        self.W_v = nn.Linear(hidden2, hidden2)
        self.drop = nn.Dropout(p=0.1)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(-1)

    def forward(self, xs, x):
        q = self.W_q(x)
        k = self.W_k(xs)
        v = self.W_v(xs)
        weight = torch.matmul(q, k.permute(0, 2, 1))
        scale = weight.size(-1) ** -0.5
        weights = self.softmax(weight * scale)
        ys = torch.matmul(self.drop(weights), v) + xs
        return ys



class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Decoder(nn.Module):
    def __init__(self, inc, dimension, outc, head, layers):
        super(Decoder, self).__init__()
        self.layers = layers
        self.act = nn.ReLU(inplace=True)
        self.act2 = nn.GELU()
        self.linear_in = nn.Linear(inc, dimension, bias=False)
        self.LN_in = nn.LayerNorm(dimension)
        self.Attention = MultiHeadAttention(h=head, d_model=dimension)
        self.linear = nn.ModuleList([nn.Linear(dimension, dimension) for _ in range(layers)])
        self.LN_fea = nn.LayerNorm(dimension)
        self.linear_out = nn.Linear(dimension, outc)

    def forward(self, x):
        x = self.act(self.linear_in(x))
        # x, atten = self.Attention(x, x, x)
        for i in range(self.layers):
            x = self.act(self.linear[i](x))
        x = self.linear_out(x)  # .view(B * N, -1)
        return x


class Covariance(nn.Module):
    def __init__(self, append_mean=False, epsilon=1e-5):
        super(Covariance, self).__init__()
        self.append_mean = append_mean
        self.epsilon = epsilon  #

    def forward(self, input):
        mean = torch.mean(input, 2, keepdim=True)
        x = input - mean.expand(-1, -1, input.size(2))
        output = torch.bmm(x, x.transpose(1, 2)) / input.size(1)

        if self.append_mean:
            mean_sq = torch.bmm(mean, mean.transpose(1, 2))
            output.add_(mean_sq)
            output = torch.cat((output, mean), 2)
            one = input.new(1, 1, 1).fill_(1).expand(mean.size(0), -1, -1)
            mean = torch.cat((mean, one), 1).transpose(1, 2)
            output = torch.cat((output, mean), 1)
        return output


class Vectorize(nn.Module):
    def __init__(self, input_size):
        super(Vectorize, self).__init__()
        row_idx, col_idx = np.triu_indices(input_size)
        self.register_buffer('row_idx', torch.LongTensor(row_idx))
        self.register_buffer('col_idx', torch.LongTensor(col_idx))

    def forward(self, input):
        output = input[:, self.row_idx, self.col_idx]
        return output


class StiefelParameter(nn.Parameter):
    """A kind of Variable that is to be considered a module parameter on the space of
        Stiefel manifold.
    """

    def __new__(cls, data=None, requires_grad=True):
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()


class SPDTransform(nn.Module):
    def __init__(self, input_size, output_size):
        super(SPDTransform, self).__init__()
        self.increase_dim = None
        self.weight = StiefelParameter(torch.FloatTensor(input_size, output_size), requires_grad=True)
        nn.init.orthogonal_(self.weight)

    def forward(self, input):
        output = input
        weight = self.weight.unsqueeze(0)
        weight = weight.expand(input.size(0), -1, -1)
        output = torch.bmm(weight.transpose(1, 2), torch.bmm(output, weight))
        return output


class SPD(nn.Module):
    def __init__(self, size, epsilon):
        super(SPD, self).__init__()
        self.rect = SPDRectified().cpu()
        self.ST_tangent = SPDTangentSpace(size, vectorize=False).cpu()
        self.epsilon = epsilon

    def add_epsilon(self, output):
        I = torch.eye(output.size(1)).expand_as(output).cpu()
        output += I * self.epsilon
        return output

    def forward(self, h_co):
        h_co = h_co.cpu()
        h_co = self.add_epsilon(h_co)
        h_vec_co_Riemannian = self.rect(h_co)
        h_hidden = self.ST_tangent(h_vec_co_Riemannian)
        return h_hidden.to("cuda")


class SPD_LeNet(nn.Module):
    def __init__(self, classes, feature_length=256, hid1=32, hid2=24, layers=3):
        super(SPD_LeNet, self).__init__()
        self.dt_joint = Feature_Joint(feature_length, feature_length, feature_length)
        self.covariance = Covariance()
        self.DS1 = SPDTransform(feature_length, hid1)
        self.DS2 = SPDTransform(1300, hid2)
        self.vec1 = Vectorize(hid1)
        self.vec2 = Vectorize(hid2)
        self.FC_down1 = nn.Linear(int((hid1 + 1) * hid1 / 2), 512)
        self.FC_down2 = nn.Linear(int((hid2 + 1) * hid2 / 2), 64)
        self.FC_combs = nn.ModuleList([nn.Linear(576, 576) for _ in range(layers)])
        self.FC_out = nn.Linear(576, classes)
        self.layers = layers
        self.act = nn.ReLU()

    def forward(self, proteins, molecules):
        Joint_Feature = self.dt_joint(proteins, molecules)
        DTI_Feature1 = self.DS1(self.covariance(Joint_Feature.permute(0, 2, 1)))
        DTI_Feature2 = self.DS2(self.covariance(Joint_Feature))
        dti_vec1 = self.vec1(DTI_Feature1)  # self.BN(dti_feature.permute(0, 2, 1))
        dti_vec2 = self.vec2(DTI_Feature2)
        dti_feature1 = self.FC_down1(dti_vec1)
        dti_feature2 = self.FC_down2(dti_vec2)
        dti_feature = torch.cat((dti_feature1, dti_feature2), 1)
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti = self.FC_out(dti_feature)
        return dti


class LeNet(nn.Module):
    def __init__(self, classes, feature_length=256, layers=3):
        super(LeNet, self).__init__()
        self.CNNs = nn.ModuleList(
            [nn.Conv1d(in_channels=feature_length, out_channels=feature_length, kernel_size=7, padding=3) for _ in
             range(layers)])
        self.BN = nn.BatchNorm1d(feature_length)  # nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])

        self.FC_combs = nn.ModuleList([nn.Linear(feature_length, feature_length) for _ in range(layers)])
        self.FC_down = nn.Linear(feature_length, 128)
        self.FC_out = nn.Linear(128, classes)
        self.layers = layers
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, proteins, molecules):
        dti_feature = torch.cat((proteins, molecules), 1)
        dti_feature = dti_feature.permute(0, 2, 1)  # self.BN(dti_feature.permute(0, 2, 1))
        for i in range(self.layers):
            dti_feature = self.act(self.CNNs[i](dti_feature)) + dti_feature
        dti_feature = dti_feature.permute(0, 2, 1)
        dti_feature = torch.mean(dti_feature, dim=1)
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti_feature = self.FC_down(dti_feature)
        dti = self.FC_out(dti_feature)
        return dti


class Feature_Joint(nn.Module):
    def __init__(self, dim_a, dim_b, out_dim):
        super(Feature_Joint, self).__init__()
        self.map = nn.Linear(dim_a + dim_b, out_dim)
        self.GELU = nn.GELU()
        self.drop = nn.Dropout(0.1)
        self.bs_attention = BS_attention(inputs=out_dim, hidden=64)

    def forward(self, feature_p, feature_d, protein_bs):
        zeros_p = torch.zeros((feature_p.shape[0], feature_p.shape[1], feature_d.shape[2])).to(feature_p.device)
        zeros_d = torch.zeros((feature_d.shape[0], feature_d.shape[1], feature_p.shape[2])).to(feature_d.device)
        feature_p_expand = torch.cat((feature_p, zeros_p), 2)
        feature_d_expand = torch.cat((zeros_d, feature_d), 2)
        # joint_feature = torch.cat((feature_p_expand, feature_d_expand), 1)
        feature_p = self.GELU(self.map(feature_p_expand))
        feature_d = self.GELU(self.map(feature_d_expand))
        joint_feature_r = torch.cat((feature_p, feature_d), 1)
        feature_a_bs = self.bs_attention(feature_p, feature_d, protein_bs)
        joint_feature_e = torch.cat((feature_a_bs, feature_d), 1)
        return self.drop(joint_feature_e), self.drop(joint_feature_r)


class BS_attention(nn.Module):
    def __init__(self, inputs=256, hidden=64):
        super(BS_attention, self).__init__()
        self.W_q = nn.Linear(inputs, hidden)
        self.W_k = nn.Linear(inputs, hidden)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, proteins, molecules, protein_bs):
        q = self.W_q(proteins)
        k = self.W_k(molecules)
        weight = torch.matmul(q, k.permute(0, 2, 1))
        weights = self.sigmoid(torch.mean(weight * protein_bs.unsqueeze(2), dim=2))
        ys = weights.unsqueeze(2) * proteins + proteins
        return ys


class Mix_classfier(nn.Module):
    def __init__(self, classes, feature_length=256, hid1=32, hid2=24, layers=3, DTA=False):
        super(Mix_classfier, self).__init__()
        self.p_l = 1200
        self.d_l = 100
        # Euc
        self.bs_attention = BS_attention(hidden=256)
        self.CNNs = nn.ModuleList(
            [nn.Conv1d(in_channels=feature_length, out_channels=feature_length, kernel_size=7, padding=3) for _ in
             range(layers)])
        self.FC_down0 = nn.Linear(256, 256)
        self.norm0 = nn.LayerNorm(256)
        # SPD
        self.dt_joint = Feature_Joint(feature_length, feature_length, feature_length)
        self.covariance = Covariance()
        self.DS1 = SPDTransform(feature_length, hid1)
        self.DS2 = SPDTransform(1300, hid2)
        self.DTI_SPD = SPD(hid1, 1e-6)
        self.DTI_SPD2 = SPD(hid2, 1e-6)
        self.vec1 = Vectorize(hid1)
        self.vec2 = Vectorize(hid2)
        self.FC_down1 = nn.Linear(int((hid1 + 1) * hid1 / 2), 256)
        self.norm1 = nn.LayerNorm(256)
        self.FC_down2 = nn.Linear(int((hid2 + 1) * hid2 / 2), 64)
        self.norm2 = nn.LayerNorm(64)
        # Comb
        
        self.classifier = SubClassifier(576, classes)
        #self.FC_combs = nn.ModuleList([nn.Linear(576, 576) for _ in range(layers)])
        #self.FC_comb_down = nn.Linear(576, 128)
        #self.FC_out = nn.Linear(128, classes)
        self.DTA = DTA
        self.layers = layers
        self.act = nn.LeakyReLU() #nn.ReLU()

    def ReSize(self, feature, N):
        weights = torch.zeros((N, self.p_l+self.d_l), device="cuda")
        for i in range(N):
            weights[i, :self.p_l] = feature[i]
        return weights.unsqueeze(2)

    def Bs_embed(self, spd_feature, protein_bs):
        spd_feature = (spd_feature + spd_feature.transpose(2, 1)) / 2
        spd_feature[:, 0:self.p_l, self.p_l:self.p_l+self.d_l] *= protein_bs.unsqueeze(2)
        spd_feature[:, self.p_l:self.p_l+self.d_l, 0:self.p_l] = spd_feature[:, 0:self.p_l, self.p_l:self.p_l+self.d_l].transpose(2, 1)
        spd_feature_out = (spd_feature + spd_feature.transpose(2, 1)) / 2
        return spd_feature_out

    def forward(self, proteins, molecules, protein_bs):
        Joint_Feature_e, Joint_Feature_r = self.dt_joint(proteins, molecules, protein_bs)
        # Euc
        dti_feature0 = Joint_Feature_e.permute(0, 2, 1)
        for i in range(self.layers):
            dti_feature0 = self.act(self.CNNs[i](dti_feature0)) + dti_feature0   
        dti_feature0 = self.FC_down0(torch.mean(dti_feature0.permute(0, 2, 1), dim=1))
        # dti_feature0, indices = torch.max(dti_feature0, dim=1)
        # SPD
        DTI_Feature1 = self.DS1(self.covariance(Joint_Feature_r.permute(0, 2, 1)))
        DTI_Feature2 = self.DS2(self.Bs_embed(self.covariance(Joint_Feature_r), protein_bs))
        DTI_Feature1 = self.DTI_SPD(DTI_Feature1)
        DTI_Feature2 = self.DTI_SPD2(DTI_Feature2)
        dti_vec1 = self.vec1(DTI_Feature1)  # self.BN(dti_feature.permute(0, 2, 1))
        dti_vec2 = self.vec2(DTI_Feature2)
        dti_feature1 = self.FC_down1(dti_vec1)
        dti_feature2 = self.FC_down2(dti_vec2)
        # Comb
        dti_feature = torch.cat((self.norm0(dti_feature0), self.norm1(dti_feature1), self.norm2(dti_feature2)), 1) #self.linear_norm()
        dti = self.classifier(dti_feature)
        # dti_feature = self.linear_norm(dti_feature)
        #for i in range(self.layers):
            #dti_feature = self.act(self.FC_combs[i](dti_feature))
        #dti_feature_out = self.act(self.FC_comb_down(dti_feature))
        #dti = self.FC_out(dti_feature_out)
        return dti

class SubClassifier(nn.Module):
    def __init__(self, input_dim, classes, hidden_dims=[576, 576, 256], dropouts=[0.05, 0.0, 0.0]):
        super(SubClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.drop1 = nn.Dropout(dropouts[0])
        
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.drop2 = nn.Dropout(dropouts[1])
        
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.drop3 = nn.Dropout(dropouts[2])
        
        self.out = nn.Linear(hidden_dims[2], classes)
        self.act = nn.LeakyReLU() # nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop1(x)

        x = self.act(self.fc2(x))
        x = self.drop2(x)

        x = self.act(self.fc3(x))
        x = self.drop3(x)

        return self.out(x)


class Mix_classfier_dta(nn.Module):
    def __init__(self, classes, feature_length=256, hid1=32, hid2=24, layers=3, DTA=False):
        super(Mix_classfier_dta, self).__init__()
        self.p_l = 1200
        self.d_l = 100
        # Euc
        self.bs_attention = BS_attention(hidden=256)
        self.CNNs = nn.ModuleList(
            [nn.Conv1d(in_channels=feature_length, out_channels=feature_length, kernel_size=7, padding=3) for _ in
             range(layers)])
        self.FC_down0 = nn.Linear(1300, 256) # p_l
        # SPD
        self.dt_joint = Feature_Joint(feature_length, feature_length, feature_length)
        self.covariance = Covariance()
        self.DS1 = SPDTransform(feature_length, hid1)
        self.DS2 = SPDTransform(1400, hid2) # p_l + d_l
        self.DTI_SPD = SPD(hid1, 1e-6)
        self.DTI_SPD2 = SPD(hid2, 1e-6)
        self.vec1 = Vectorize(hid1)
        self.vec2 = Vectorize(hid2)
        self.FC_down1 = nn.Linear(int((hid1 + 1) * hid1 / 2), 256)
        self.FC_down2 = nn.Linear(int((hid2 + 1) * hid2 / 2), 64)
        # Comb
        self.linear_norm = nn.LayerNorm(576)
        self.FC_combs = nn.ModuleList([nn.Linear(576, 576) for _ in range(layers)])
        self.FC_comb_down = nn.Linear(576, 512)
        self.FC_out = nn.Linear(512, classes)
        self.layers = layers
        self.act = nn.ReLU()

    def ReSize(self, feature, N):
        weights = torch.zeros((N, self.p_l+self.d_l), device="cuda")
        for i in range(N):
            weights[i, :self.p_l] = feature[i]
        return weights.unsqueeze(2)

    def Bs_embed(self, spd_feature, protein_bs):
        spd_feature = (spd_feature + spd_feature.transpose(2, 1)) / 2
        spd_feature[:, 0:self.p_l, self.p_l:self.p_l+self.d_l] *= protein_bs.unsqueeze(2)
        spd_feature[:, self.p_l:self.p_l+self.d_l, 0:self.p_l] = spd_feature[:, 0:self.p_l, self.p_l:self.p_l+self.d_l].transpose(2, 1)
        spd_feature_out = (spd_feature + spd_feature.transpose(2, 1)) / 2
        return spd_feature_out

    def forward(self, proteins, molecules, protein_bs):
        Joint_Feature_e, Joint_Feature_r = self.dt_joint(proteins, molecules, protein_bs)
        # Euc
        dti_feature0 = Joint_Feature_e.permute(0, 2, 1)
        for i in range(self.layers):
            dti_feature0 = self.act(self.CNNs[i](dti_feature0)) + dti_feature0
        dti_feature0 = torch.mean(dti_feature0.permute(0, 2, 1), dim=1)
        # SPD
        DTI_Feature1 = self.DS1(self.covariance(Joint_Feature_r.permute(0, 2, 1)))
        DTI_Feature2 = self.DS2(self.Bs_embed(self.covariance(Joint_Feature_r), protein_bs))
        DTI_Feature1 = self.DTI_SPD(DTI_Feature1)
        DTI_Feature2 = self.DTI_SPD2(DTI_Feature2)
        dti_vec1 = self.vec1(DTI_Feature1)  # self.BN(dti_feature.permute(0, 2, 1))
        dti_vec2 = self.vec2(DTI_Feature2)
        dti_feature1 = self.FC_down1(dti_vec1)
        dti_feature2 = self.FC_down2(dti_vec2)
        # Comb
        dti_feature = torch.cat((dti_feature0, dti_feature1, dti_feature2), 1)
        dti_feature = self.linear_norm(dti_feature)
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dta_feature_out = self.act(self.FC_comb_down(dti_feature))
        dta = self.FC_out(dta_feature_out)
        return dta


class DTFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, k=2.0):
        super(DTFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.k = k

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:, 1]
        targets = targets.float()

        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = -alpha_weight * focal_weight * torch.log(pt + 1e-9)

        positive_loss = loss[targets == 1]
        negative_loss = loss[targets == 0]

        if len(positive_loss) > 0:
            mean_positive_loss = positive_loss.mean()
            std_positive_loss = positive_loss.std()
        else:
            mean_positive_loss = torch.tensor(0.0, device=logits.device)
            std_positive_loss = torch.tensor(0.0, device=logits.device)

        threshold = mean_positive_loss + self.k * std_positive_loss

        if len(negative_loss) > 0:
            filtered_negative_loss = torch.where(
                negative_loss < threshold, negative_loss, torch.tensor(0.0, device=negative_loss.device)
            )
        else:
            filtered_negative_loss = torch.tensor(0.0, device=logits.device)

        if positive_loss.numel() == 0 and filtered_negative_loss.numel() == 0:
            total_loss = torch.tensor(0.0, device=logits.device)
        elif positive_loss.numel() == 0:
            total_loss = filtered_negative_loss
        elif filtered_negative_loss.numel() == 0:
            total_loss = positive_loss
        else:
            total_loss = torch.cat([positive_loss, filtered_negative_loss])

        return total_loss.mean()


class DTL(nn.Module):
    def __init__(self, k=1.0):
        super(DTL, self).__init__()
        self.k = k

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:, 1]
        targets = targets.float()
        loss = -targets * torch.log(probs + 1e-9) - (1 - targets) * torch.log(1 - probs + 1e-9)
        positive_loss = loss[targets == 1]
        negative_loss = loss[targets == 0]
        if len(positive_loss) > 0:
            mean_positive_loss = positive_loss.mean()
            std_positive_loss = positive_loss.std()
        else:
            mean_positive_loss = torch.tensor(0.0, device=logits.device)
            std_positive_loss = torch.tensor(0.0, device=logits.device)
        threshold = mean_positive_loss + self.k * std_positive_loss
        if len(negative_loss) > 0:
            filtered_negative_loss = torch.where(
                negative_loss < threshold, negative_loss, torch.tensor(0.0, device=negative_loss.device)
            )
        else:
            filtered_negative_loss = torch.tensor(0.0, device=logits.device)
        total_loss = torch.cat([positive_loss, filtered_negative_loss])
        return total_loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        probs = F.softmax(inputs, dim=1)  # Shape: (batch_size, num_classes)

        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))  # Shape: (batch_size, num_classes)
        targets_one_hot = targets_one_hot.type_as(inputs)

        probs = torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)
        pt = torch.sum(probs * targets_one_hot, dim=1)  # Shape: (batch_size,)

        focal_weight = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_weight * torch.log(pt)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
            

class PolyLoss(nn.Module):
    def __init__(self, epsilon=1.0, reduction='mean'):
        super(PolyLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        # self.CELoss = FocalLoss(alpha=0.6, gamma=2.0, reduction='mean') #nn.CrossEntropyLoss(weight=torch.FloatTensor([0.4, 0.6]).cuda(), reduction='none')
        # self.CELoss2 = nn.CrossEntropyLoss(label_smoothing=0.05, reduction='none')

    def forward(self, logits, targets):
        """
        logits: Tensor of shape (B, C)
        targets: Tensor of shape (B,) with class indices
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        # ce_loss = self.CELoss(logits, targets)
        probs = F.softmax(logits, dim=1)
        pt = probs[torch.arange(logits.size(0)), targets]  # p_t for each sample

        poly_loss = ce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            return poly_loss.mean()
        elif self.reduction == 'sum':
            return poly_loss.sum()
        else:
            return poly_loss