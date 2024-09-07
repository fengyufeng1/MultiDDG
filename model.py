import pickle
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch.nn import AdaptiveMaxPool1d
from torch.optim.lr_scheduler import StepLR

# Seed
SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

# SeqVec, ProtBert, ProtT5, ESM2_t33(1280+17=1297)
Protein_model = 'SeqVec'

C_INPUT_DIM = 17
G_INPUT_DIM = 1024
A_INPUT_DIM = 15

C_HIDDEN_DIM = 64
G_HIDDEN_DIM = 128 
A_HIDDEN_DIM = 64

G_HEADS = 8
A_HEADS = 2

DROPOUT = 0.1

LEARNING_RATE = 5E-4
WEIGHT_DECAY = 0
BATCH_SIZE = 1
NUMBER_EPOCHS = 50

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda:0") 

def get_model_feature(sequence_name, path):
    model_feature = np.load(path + sequence_name + '.npy')
    return model_feature.astype(np.float32)

def load_graph(sequence_name, path):
    norm_matrix = np.load(path + sequence_name + ".npy")
    return norm_matrix

class ProDataset(Dataset):
    def __init__(self, feature_path, dataframe):
        self.names = dataframe['ID'].values
        self.WT_names = dataframe['WT_ID'].values
        self.labels = dataframe['label'].values
        self.feature_path = feature_path

    def __getitem__(self, index):
        PDB_name = self.names[index]
        WT_PDB_name = self.WT_names[index]
        label = np.array(self.labels[index])

        # node and edge
        node = get_model_feature(PDB_name, self.feature_path + f"Feature_{Protein_model}/")
        wt_node = get_model_feature(WT_PDB_name, self.feature_path + f"Feature_{Protein_model}/")

        graph = load_graph(PDB_name, self.feature_path + f"Edge_index/")
        wt_graph = load_graph(WT_PDB_name, self.feature_path + f"Edge_index/")

        atom = get_model_feature(PDB_name, self.feature_path + f"Atom_feature/")
        wt_atom = get_model_feature(WT_PDB_name, self.feature_path + f"Atom_feature/")

        edge = load_graph(PDB_name, self.feature_path + f"Atom_edge_index/")
        wt_edge = load_graph(WT_PDB_name, self.feature_path + f"Atom_edge_index/")

        return PDB_name, WT_PDB_name, label, node, graph, wt_node, wt_graph, atom, edge, wt_atom, wt_edge

    def __len__(self):
        return len(self.labels)

"""
Model_code
"""
class Ami_AttGNN(nn.Module):
    def __init__(self, g_heads, g_nfeat, g_nhidden, dropout):
        super(Ami_AttGNN, self).__init__()

        self.conv1 = GATConv(g_nfeat, 128, heads=g_heads, dropout=dropout*2)
        self.fc1 = nn.Linear(128*g_heads, g_nhidden)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, wt_x, wt_adj):
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        wt_batch = torch.zeros(wt_x.shape[0], dtype=torch.long, device=wt_x.device)

        # MT
        x = self.conv1(x, adj)
        x = self.relu(x)

        x = gep(x, batch)    
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        # WT
        wt_x = self.conv1(wt_x, wt_adj)
        wt_x = self.relu(wt_x)

        wt_x = gep(wt_x, wt_batch) 
        wt_x = self.relu(self.fc1(wt_x))
        wt_x = self.dropout(wt_x)

        xd = x - wt_x
        xc = torch.cat((x, wt_x, xd), 1)  
        
        return xc

class Atom_AttGNN(nn.Module):
    def __init__(self, a_heads, a_nfeat, a_nhidden, dropout):
        super(Atom_AttGNN, self).__init__()

        self.conv1 = GATConv(a_nfeat, 64, heads=a_heads, dropout=dropout)
        self.fc1 = nn.Linear(64*a_heads, a_nhidden)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, wt_x, wt_adj):
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        wt_batch = torch.zeros(wt_x.shape[0], dtype=torch.long, device=wt_x.device)

        # MT
        x = self.conv1(x, adj)
        x = self.relu(x)

        x = gep(x, batch)      
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        # WT
        wt_x = self.conv1(wt_x, wt_adj)
        wt_x = self.relu(wt_x)

        wt_x = gep(wt_x, wt_batch) 
        wt_x = self.relu(self.fc1(wt_x))
        wt_x = self.dropout(wt_x)

        xd = x - wt_x
        xc = torch.cat((x, wt_x, xd), 1)  

        return xc

class NCNN(nn.Module):
    def __init__(self, c_nfeat, c_nhidden, dropout):
        super(NCNN, self).__init__()
        self.kernel_size = 3
        self.padding = 1
        self.output_size_1 = 8
        self.output_size_2 = 4

        self.conv1 = nn.Conv1d(in_channels=c_nfeat, out_channels=64, kernel_size=self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=self.kernel_size, padding=self.padding)

        self.adaptive_pool1 = nn.AdaptiveMaxPool1d(output_size=self.output_size_1)
        self.adaptive_pool2 = nn.AdaptiveMaxPool1d(output_size=self.output_size_2)

        self.fc1 = nn.Linear(in_features=32 * self.output_size_2, out_features=c_nhidden)
        
    def forward(self, x, wt_x):
        # MT
        x = x.unsqueeze(0)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = F.relu(x)

        x = self.adaptive_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = self.adaptive_pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        # WT
        wt_x = wt_x.unsqueeze(0)
        wt_x = wt_x.transpose(1, 2) 
        wt_x = self.conv1(wt_x)
        wt_x = F.relu(wt_x)

        wt_x = self.adaptive_pool1(wt_x)
        wt_x = self.conv2(wt_x)
        wt_x = F.relu(wt_x)

        wt_x = self.adaptive_pool2(wt_x)
        wt_x = wt_x.view(wt_x.size(0), -1)
        wt_x = self.fc1(wt_x)

        xd = x - wt_x
        xc = torch.cat((x, wt_x, xd), 1)  

        return xc
    
class MLP(nn.Module):
    def __init__(self, dropout):
        super(MLP, self).__init__()
        self.fcs1 = nn.Linear(256+256+256, 256)
        self.relu1 = nn.ReLU()
        self.fcs2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU()
        self.fcs3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out = self.fcs1(x)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.fcs2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.fcs3(out)
        out = out.squeeze()

        return out    

class GraphPPIS(nn.Module):
    def __init__(self, c_nfeat, c_nhidden, g_heads, g_nfeat, g_nhidden, a_heads, a_nfeat, a_nhidden, dropout):
        super(GraphPPIS, self).__init__()

        # Model
        self.cnn = NCNN(c_nfeat=c_nfeat, c_nhidden=c_nhidden, dropout=dropout)

        self.ami_attgcn = Ami_AttGNN(g_heads = g_heads, g_nfeat = g_nfeat, g_nhidden = g_nhidden, dropout = dropout)

        self.atom_attgcn = Atom_AttGNN(a_heads = a_heads, a_nfeat = a_nfeat, a_nhidden = a_nhidden, dropout = dropout)

        # MLP
        self.Protein_MLP = MLP(dropout = dropout)
         
        self.criterion = nn.SmoothL1Loss(reduction='sum')

        self.optimizer = torch.optim.Adam(self.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

        
    def forward(self, x, adj, wt_x, wt_adj, atom, edge, wt_atom, wt_edge):       

        ami_gcn = self.ami_attgcn(x[:, :1024], adj, wt_x[:, :1024], wt_adj) 

        atom_gcn = self.atom_attgcn(atom, edge, wt_atom, wt_edge)

        struc_cnn = self.cnn(x[:, -17:], wt_x[:, -17:])

        out = torch.cat((ami_gcn, atom_gcn, struc_cnn), dim=1)

        out = self.Protein_MLP(out)

        return out


