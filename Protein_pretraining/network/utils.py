import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from network.MultiHeadAttention import MultiHeadAttention, Attention_ADJ
import numpy as np

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
        a_input = torch.matmul(h_qk, h_qk.permute(0,2,1))
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
        x = self.linear_out(x)#.view(B * N, -1)
        return x

class Covariance(nn.Module):
    def __init__(self, append_mean=False, epsilon=1e-5):
        super(Covariance, self).__init__()
        self.append_mean = append_mean
        self.epsilon = epsilon  # 小的常数用于添加到协方差矩阵的对角线上

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
        output = torch.bmm(weight.transpose(1,2), torch.bmm(output, weight))
        return output

class Feature_Joint(nn.Module):
    def __init__(self, dim_a, dim_b, out_dim):
        super(Feature_Joint, self).__init__()
        self.map = nn.Linear(dim_a + dim_b, out_dim)
        self.GELU = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, feature_a, feature_b):
        zeros_a = torch.zeros((feature_a.shape[0], feature_a.shape[1], feature_b.shape[2])).to(feature_a.device)
        zeros_b = torch.zeros((feature_b.shape[0], feature_b.shape[1], feature_a.shape[2])).to(feature_b.device)
        feature_a_expand = torch.cat((feature_a, zeros_a), 2)
        feature_b_expand = torch.cat((zeros_b, feature_b), 2)
        joint_feature = torch.cat((feature_a_expand, feature_b_expand), 1)
        joint_feature = self.GELU(self.map(joint_feature))
        return self.drop(joint_feature)

class SPD_LeNet(nn.Module):
    def __init__(self, classes, feature_length=256, hid1=32, hid2=24, layers=3):
        super(SPD_LeNet, self).__init__()
        self.dt_joint = Feature_Joint(feature_length, feature_length, feature_length)
        self.covariance = Covariance()
        self.DS1 = SPDTransform(feature_length, hid1)
        self.DS2 = SPDTransform(1300, hid2)
        self.vec1 = Vectorize(hid1)
        self.vec2 = Vectorize(hid2)
        self.FC_down1 = nn.Linear(int((hid1+1)*hid1/2), 512)
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
        self.dt_joint = Feature_Joint(feature_length, feature_length, feature_length)
        self.CNNs = nn.ModuleList(
            [nn.Conv1d(in_channels=feature_length, out_channels=feature_length, kernel_size=7, padding=3) for _ in range(layers)])
        self.BN = nn.BatchNorm1d(feature_length)  # nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])

        self.FC_combs = nn.ModuleList([nn.Linear(feature_length, feature_length) for _ in range(layers)])
        self.FC_down = nn.Linear(feature_length, 128)
        self.FC_out = nn.Linear(128, classes)
        self.layers = layers
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, proteins, molecules):
        dti_feature = self.dt_joint(proteins, molecules)
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

class Mix_classfier(nn.Module):
    def __init__(self, classes, feature_length=256, hid1=32, hid2=24, layers=3):
        super(Mix_classfier, self).__init__()
        # Euc
        self.CNNs = nn.ModuleList(
            [nn.Conv1d(in_channels=feature_length, out_channels=feature_length, kernel_size=7, padding=3) for _ in range(layers)])
        self.FC_down0 = nn.Linear(feature_length, 128)
        # SPD
        self.dt_joint = Feature_Joint(feature_length, feature_length, feature_length)
        self.covariance = Covariance()
        self.DS1 = SPDTransform(feature_length, hid1)
        self.DS2 = SPDTransform(1300, hid2)
        self.vec1 = Vectorize(hid1)
        self.vec2 = Vectorize(hid2)
        self.FC_down1 = nn.Linear(int((hid1+1)*hid1/2), 320)
        self.FC_down2 = nn.Linear(int((hid2 + 1) * hid2 / 2), 64)
        # Comb
        self.FC_combs = nn.ModuleList([nn.Linear(512, 512) for _ in range(layers)])
        self.FC_down = nn.Linear(512, 128)
        self.FC_out = nn.Linear(128, classes)
        self.layers = layers
        self.act = nn.ReLU()

    def forward(self, proteins, molecules):
        Joint_Feature = self.dt_joint(proteins, molecules)
        # Euc
        dti_feature0 = Joint_Feature.permute(0, 2, 1)
        for i in range(self.layers):
            dti_feature0 = self.act(self.CNNs[i](dti_feature0)) + dti_feature0
        dti_feature0 = self.FC_down0(torch.mean(dti_feature0.permute(0, 2, 1), dim=1))
        # SPD
        DTI_Feature1 = self.DS1(self.covariance(Joint_Feature.permute(0, 2, 1)))
        DTI_Feature2 = self.DS2(self.covariance(Joint_Feature))
        dti_vec1 = self.vec1(DTI_Feature1)  # self.BN(dti_feature.permute(0, 2, 1))
        dti_vec2 = self.vec2(DTI_Feature2)
        dti_feature1 = self.FC_down1(dti_vec1)
        dti_feature2 = self.FC_down2(dti_vec2)
        # Comb
        dti_feature = torch.cat((dti_feature0, dti_feature1, dti_feature2), 1)
        # dti_feature = self.linear_norm(dti_feature)
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti_feature = self.act(self.FC_down(dti_feature))
        dti = self.FC_out(dti_feature)
        return dti
