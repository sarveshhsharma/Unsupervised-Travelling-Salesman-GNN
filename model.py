import torch
import torch.nn.functional as F
from torch import tensor
import torch.nn
from feature_extraction import GCN_diffusion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn

class GCNConv(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.a = nn.Parameter(torch.zeros(size=(2 * hidden_dim, 1)))

    def forward(self, X, adj, moment=1, device='cuda'):
        """
        Params
        ------
        adj [batch x nodes x nodes]: adjacency matrix
        X [batch x nodes x features]: node features matrix
        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        support0 = X
        B = support0.size(0) # batchsize
        N = support0.size(1) # n
        h = support0
        gcn_diffusion_list = GCN_diffusion(adj, 3, support0)
        h_A = gcn_diffusion_list[0]
        h_A2 = gcn_diffusion_list[1]
        h_A3 = gcn_diffusion_list[2]

        h_A = nn.LeakyReLU()(h_A)
        h_A2 = nn.LeakyReLU()(h_A2)
        h_A3 = nn.LeakyReLU()(h_A3)

        a_input_A = torch.cat((h, h_A), dim=2).unsqueeze(1)
        a_input_A2 = torch.cat((h, h_A2), dim=2).unsqueeze(1)
        a_input_A3 = torch.cat((h, h_A3), dim=2).unsqueeze(1) # [b,1,n,2f]

        a_input = torch.cat((a_input_A, a_input_A2,a_input_A3), 1).view(B,3,N,-1)

        e = torch.matmul(nn.functional.relu(a_input), self.a).squeeze(3)
        attention = F.softmax(e, dim=1).view(B,3, N, -1)
        h_all = torch.cat((h_A.unsqueeze(dim=1), h_A2.unsqueeze(dim=1),h_A3.unsqueeze(dim=1))).view(B, 3,N,-1)
        h_prime = torch.mul(attention, h_all)
        h_prime = torch.mean(h_prime, 1) # (B,n,f)
        X = self.linear1(h_prime)
        X = F.leaky_relu(X)
        X = self.linear2(X)
        X = F.leaky_relu(X)
        return X

class GNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.input_dim = input_dim
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GCNConv(hidden_dim))

        self.mlp1 = nn.Linear(hidden_dim * (1 + n_layers), hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.m = nn.Softmax(dim=1)

    def forward(self, X, adj, moment=1):
        X = self.in_proj(X)
        hidden_states = X
        for layer in self.convs:
            X = layer(X, adj, moment=moment)
            hidden_states = torch.cat([hidden_states, X], dim=-1)

        X = hidden_states
        X = self.mlp1(X)
        X = F.leaky_relu(X)
        X = self.mlp2(X)
        X = self.m(X)
        return X