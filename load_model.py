import torch
import torch.nn.functional as F
from torch.nn import Linear
import time
from torch import tensor
import torch.nn
from utils import TSPLoss,edge_overlap,get_heat_map
import pickle
from torch.utils.data import  Dataset,DataLoader# use pytorch dataloader
from random import shuffle
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--num_of_nodes', type=int, default=100, help='Graph Size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning Rate')
parser.add_argument('--smoo', type=float, default=0.1,
                    help='smoo')
parser.add_argument('--moment', type=int, default=1,
                    help='scattering moment')
parser.add_argument('--hidden', type=int, default=64, #64
                    help='Number of hidden units.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch_size')
parser.add_argument('--nlayers', type=int, default=3, #3
                    help='num of layers')
parser.add_argument('--use_smoo', action='store_true')
parser.add_argument('--EPOCHS', type=int, default=300,
                    help='epochs to train')
parser.add_argument('--topk', type=int, default=20, #20
                    help='top k elements per row, should equal to int Rec_Num = 20 in Search/code/include/TSP_IO.h')
parser.add_argument('--penalty_coefficient', type=float, default=2.,
                    help='penalty_coefficient')
parser.add_argument('--wdecay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--temperature', type=float, default=2.,
                    help='temperature for adj matrix')
parser.add_argument('--diag_penalty', type=float, default=3.,
                    help='penalty on the diag')
parser.add_argument('--rescale', type=float, default=1.,
                    help='rescale for xy plane')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device')
args = parser.parse_args()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.cuda.manual_seed(args.seed)
device = args.device


tsp_instances = np.load('data/test_tsp_instance_%d.npy'%args.num_of_nodes) # 128 instances
NumofTestSample = tsp_instances.shape[0]
Std = np.std(tsp_instances, axis=1)
Mean = np.mean(tsp_instances, axis=1)
#print(tsp_instances.shape) = (10000, 100, 2)
tsp_instances = tsp_instances - Mean.reshape((NumofTestSample,1,2))
tsp_instances = args.rescale * tsp_instances # 2.0 is the rescale

tsp_sols = np.load('data/test_tsp_sol_%d.npy'%args.num_of_nodes)
# print(tsp_sols.shape) = (10000, 101) 
# contains the array if started from 0(origin) which all city(index wise to visit)
# print(tsp_instances[0])
# print(tsp_sols[0])


total_samples = tsp_instances.shape[0]
import json

from model import GNN
#scattering model
model = GNN(input_dim=2, hidden_dim=args.hidden, output_dim=args.num_of_nodes, n_layers=args.nlayers)
#model = model.to(device)
from scipy.spatial import distance_matrix
#So this line builds a GNN with:
# 2D inputs per node (coordinates),
# hidden processing units,
# and outputs for each of the num_of_nodes (TSP nodes). = 100
# from scipy.spatial import distance_matrix = function to compute pairwise distances between points in Euclidean space.


### count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Total number of parameters:')
print(count_parameters(model)) # = 6464



def coord_to_adj(coord_arr):
    dis_mat = distance_matrix(coord_arr,coord_arr)
    return dis_mat
#coord_arr: A 2D NumPy array of shape (N, 2), where N is the number of nodes (cities), and each row represents (x, y) coordinates of a node.
#coord_arr, this creates an N × N matrix (dis_mat) where dis_mat[i][j] is the distance between node i and node j.
#here nodes = city 


tsp_instances_adj = np.zeros((total_samples,args.num_of_nodes,args.num_of_nodes)) #(10000,100,100) for each instances NxN matrix made
for i in range(total_samples):
    tsp_instances_adj[i] = coord_to_adj(tsp_instances[i])


class TSP_Dataset(Dataset):
    def __init__(self, coord,data, targets):
        # coord = tsp_instances (10000,100,2)
        # data = tsp_instances_adj (10000,100,100)
        # targets = tsp_sols
        #converting your input data (which is likely NumPy arrays or Python lists) into PyTorch tensors
        self.coord = torch.FloatTensor(coord)
        self.data = torch.FloatTensor(data)
        self.targets = torch.LongTensor(targets)

    def __getitem__(self, index):
        xy_pos = self.coord[index] 
        #self.coord shape is (num_samples, num_cities, 2)
        x = self.data[index]
        #index-th sample of data, which could be the adjacency/distance matrix for that instance.
        y = self.targets[index]
        #solution for a particular x
#        tsp_instance = Data(coord=x,sol=y)
        return tuple(zip(xy_pos,x,y))
        #zip() is a built-in Python function that takes multiple sequences (like lists or tuples) and “zips” them together into pairs (or tuples
        # a = [1, 2, 3]
        # b = ['a', 'b', 'c']

        # zipped = zip(a, b)
        # print(tuple(zipped))
        # ((1, 'a'), (2, 'b'), (3, 'c')) = output
    def __len__(self):
        return len(self.data) # gives 10000 = number of instances (number of samples)




dataset = TSP_Dataset(tsp_instances,tsp_instances_adj,tsp_sols)
testdata = dataset[0:] ##this is very important!
# if written testdata = dataset ## object of dataset 
# dataset[0:] = retrives touple zip(xy_pos,x,y)


TestData_size = len(testdata) #number of samples (10000)

batch_size = args.batch_size
test_loader = DataLoader(testdata, batch_size, shuffle=False)

#avoids shuffling of data
mask = torch.ones(args.num_of_nodes, args.num_of_nodes)
#Creates a square matrix of shape (num_nodes, num_nodes) = 100,100
mask.fill_diagonal_(0)
#Sets the diagonal elements to 0.

print('Finish Inference!')

def test(loader,topk = 20): #20
    #here loader = test_loader = some part of testdata
    #how many top predicted nodes per node you want to keep.
    avg_size = 0
    total_cost = 0.0
    full_edge_overlap_count = 0

    TestData_size = len(loader.dataset) # 10000
    Saved_indices = np.zeros((TestData_size,args.num_of_nodes,topk))
    #(10000,100,20)
    #top-k predicted node indices for each node, per instance.
    Saved_Values = np.zeros((TestData_size,args.num_of_nodes,topk))
    # corresponding values (e.g., probabilities or scores) of those top-k nodes.
    #Saved_indices[0] is a 2D array of shape (100, 20) for the first instance (the 0-th TSP problem).
    #For each city/node i (0 to 99) in that first instance, you have a list of top 20 predicted nodes related to city i.
    Saved_sol = np.zeros((TestData_size,args.num_of_nodes+1)) 
    #the ground truth solutions for each instance (length = num_of_nodes + 1, probably TSP tour with return to start).
    Saved_pos = np.zeros((TestData_size,args.num_of_nodes,2))
    #positions (coordinates) of the nodes.
    count = 0
    model.eval()
    for batch in loader: #batch = (tsp_instances,tsp_instances_adj,tsp_sols)
        batch_size = batch[0].size(0)
        xy_pos = batch[0]
        #for one sample whole xy (100,2 shape)
        distance_m = batch[1]
        #adjacency matrix shape = 100,100
        sol = batch[2]
        #solutions shape = (100,) = order
        adj = torch.exp(-1.*distance_m/args.temperature)
        # adj(ij) = e^ -(distance(ij)/temperature(default = 2))
        adj *= mask
        #Multiplying adj by mask sets the diagonal elements (self-connections) to 0, no selfloop

        # start here:
        t0 = time.time()
        output = model(xy_pos,adj) 
        #making predictions
        # model has learned parameters (weights) from training that define how it computes those heat maps based on input node coordinates and adjacency.
        #model takes whole 100,2 matrix having 100 city coordinates and adjacency matrix (100,100)


        t1 = time.time()
        Heat_mat = get_heat_map(SctOutput=output,num_of_nodes=args.num_of_nodes,device = device)
        print('It takes %.5f seconds from instance: %d to %d'%(t1 - t0,count,count + batch_size))
        #formats a float with 5 digits after the decimal point
        #denotes time taken for the calculation
        sol_indicies = torch.topk(Heat_mat,topk,dim=2).indices
        #selects the top k neighbors (with highest "heat"/score) out of num_of_nodes
        sol_values = torch.topk(Heat_mat,topk,dim=2).values
        #sol_indices[i][j]: Top-k neighbor indices with strongest edge weights (after transformation).
        #sol_values[i][j]: Their corresponding scores.


#        print(sol_values.size())
#        print(batch_size)
        Saved_indices[count:batch_size+count] = sol_indicies.detach().cpu().numpy()
        Saved_Values[count:batch_size+count] = sol_values.detach().cpu().numpy()
        Saved_sol[count:batch_size+count] = sol.detach().cpu().numpy()
        Saved_pos[count:batch_size+count] = xy_pos.detach().cpu().numpy()
        count = count + batch_size
        # Saved_indices[i]: top-k neighbors of all nodes in the i-th graph.
        # Saved_Values[i]: corresponding scores.
        # Saved_sol[i]: actual tour.
        # Saved_pos[i]: city coordinates.


    return Saved_indices,Saved_Values,Saved_sol,Saved_pos,Heat_mat

# to load a predefined model
# TSP 100

model_name = 'Saved_models/TSP_%d/scatgnn_layer_%d_hid_%d_model_20_temp_3.500.pth'%(args.num_of_nodes,args.nlayers,args.hidden)# topk = 20
model.load_state_dict(torch.load(model_name)) #for 20 epoch
Saved_indices,Saved_Values,Saved_sol,Saved_pos,Heat_mat = test(test_loader,topk = args.topk) # epoch=20>10


import numpy as np

Q = Saved_pos
A = Saved_sol 
C = Saved_indices
V = Saved_Values

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def compute_tour_length(tour, coords):
    return sum(euclidean_distance(coords[int(tour[i])], coords[int(tour[(i + 1) % len(tour)])]) for i in range(len(tour)))

def build_greedy_tour(pred_neighbors, coords):
    num_nodes = coords.shape[0]
    visited = [False] * num_nodes
    tour = [0]  # Start at 0
    visited[0] = True

    # Visit all other nodes
    for _ in range(num_nodes - 1):
        last = tour[-1]
        found = False
        for nxt in pred_neighbors[int(last)]:  # Try top-k neighbors
            if not visited[int(nxt)]:
                tour.append(nxt)
                visited[int(nxt)] = True
                found = True
                break
        if not found:  # Fallback if no valid neighbor
            unvisited = [j for j in range(num_nodes) if not visited[j]]
            if unvisited:
                tour.append(unvisited[0])
                visited[unvisited[0]] = True

    tour.append(0)  # Explicitly end at 0
    return tour

def two_opt(tour, coords):
    # Ensure tour starts/ends with 0
    if tour[0] != 0 or tour[-1] != 0:
        tour = [0] + [n for n in tour if n != 0] + [0]
    
    improved = True
    best_distance = compute_tour_length(tour, coords)
    n = len(tour)
    
    while improved:
        improved = False
        for i in range(1, n-2):  # Skip fixed endpoints
            for j in range(i+1, n-1):
                if j - i == 1: continue
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                new_distance = compute_tour_length(new_tour, coords)
                if new_distance < best_distance:
                    tour = new_tour
                    best_distance = new_distance
                    improved = True
    return tour

# Q = Saved_pos, C = Saved_indices
# For test check for instance 0
instance_id = 0
coords = Q[instance_id]
pred_neighbors = C[instance_id]

# Build and refine tour (automatically starts/ends at 0)
init_tour = build_greedy_tour(pred_neighbors, coords)
opt_tour = two_opt(init_tour, coords)
opt_tour = [int(x) for x in opt_tour]
# Print or return the tour
print("Heat MAP: ",Heat_mat)
print("Predicted tour:", opt_tour)
print("Length:", compute_tour_length(opt_tour, coords))

gaps = []
for i in range(100):
    # Get reference solution and coordinates
    ref_tour = tsp_sols[i].astype(int)  # Ensure integer indices
    coords = Q[i]
    
    # Calculate reference length (optimal)
    optimal_length = compute_tour_length(ref_tour, coords)
    
    # Generate your prediction
    init_tour = build_greedy_tour(C[i], coords)
    pred_tour = two_opt(init_tour, coords)
    
    # Calculate predicted length
    pred_length = compute_tour_length(pred_tour, coords)
    
    # Compute gap
    gap = (pred_length - optimal_length) / optimal_length
    gaps.append(gap)

# Convert to percentage gaps
gaps = np.array(gaps) * 100
# Average optimality gap
mean_gap = np.mean(gaps)

# Median gap
median_gap = np.median(gaps)

# Percentage of solutions within 5% of optimal
within_5pct = np.mean(gaps <= 5) * 100

# Worst-case performance
max_gap = np.max(gaps)
print(f"Average optimality gap: {mean_gap:.2f}%")
print(f"Median optimality gap: {median_gap:.2f}%")
print(f"Solutions within 5% of optimal: {within_5pct:.1f}%")
print(f"Maximum gap: {max_gap:.2f}%")
