import torch
import numpy as np

def GCN_diffusion(W,order,feature):
    """
    W: [batchsize,n,n]
    feature: [batchsize,n,n]
    """
    identity_matrices = torch.eye(W.size(1)).repeat(W.size(0), 1, 1)
    I_n = identity_matrices
    A_gcn = W + I_n #[b,n,n]
    ###
    degrees = torch.sum(A_gcn,2)
    degrees = degrees.unsqueeze(dim=2) # [b,n,1]
    D = degrees
    ##
    D = torch.pow(D, -0.5)
    gcn_diffusion_list = []
    A_gcn_feature = feature
    for i in range(order):
        A_gcn_feature = D*A_gcn_feature
        A_gcn_feature = torch.matmul(A_gcn,A_gcn_feature) # batched matrix x batched matrix https://pytorch.org/docs/stable/generated/torch.matmul.html
        A_gcn_feature = torch.mul(A_gcn_feature,D)
        gcn_diffusion_list += [A_gcn_feature,]
    return gcn_diffusion_list




