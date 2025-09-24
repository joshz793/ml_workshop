import sys
import json
import networkx as nx
import torch
import numpy as np
from smiletovectors import get_full_neighbor_vectors
from paulingelectro import get_eleneg_diff_mat
from rdkit import Chem
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

au2eV = 27.21139
EN_MAT = get_eleneg_diff_mat()

### Helper functions from hackathon
def load_data_from_file(filename) -> dict:
    """
    Load a dictionary of graphs from JSON file.
    """
    with open(filename, "r") as file_handle:
        string_dict = json.load(file_handle)
    return _load_data_from_string_dict(string_dict)

def _load_data_from_string_dict(string_dict) -> dict:
	result_dict = {}
	for key in string_dict:
		graph = nx.node_link_graph(string_dict[key], edges="edges")
		result_dict[key] = graph
	return result_dict

### Custom functions to get orbital energies of lone atoms
def get_l(l):
    '''
    l = s, p, d, f, g
    '''
    return {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g':4,}.get(l, "l value is not valid")

def get_n_l(orb):
    n = int(orb[0])
    l = get_l(str(orb[1]))
    return n, l

def giveorbitalenergy(ele, orb):
    with open('orbitalenergy.json', 'r') as f:
        data = json.load(f)
    try:
        orbenegele = data[ele]
        del data
    except KeyError:
        raise KeyError("Element symbol not found")
    
    n, l = get_n_l(orb)
    cbenergy = orbenegele[str(l)][n-l-1]
    cbenergy *= au2eV
    return cbenergy

def get_lmat_vec(symbol, num_elements=100):
    '''
    Generates a row of the "left matrix" for the electronegativity environment calculation.
        - The left matrix is a zero-matrix of size n_samples x num_elements.
        - Each row has 1 non-zero element at the index representing the atomic number - 1
        - e.g.: a lmat representing Carbon will have a "1" at index 5, with "0" everywhere else
    '''
    lmat = np.zeros(num_elements).reshape(1,-1)
    lmat[0, Chem.Atom(symbol).GetAtomicNum() - 1] = 1
    return lmat

def get_eneg_mats(smile, num_elements=100):
    '''
    Generates a "left matrix" and "connectivity matrix" for a given SMILE string.
        - The left matrix is a zero-matrix of size n_samples x num_elements.
        - Each row has 1 non-zero element at the index representing the atomic number - 1
        - e.g.: a lmat row representing Carbon will have a "1" at index 5, with "0" everywhere else
    Returns: (lmat, cmat)
        - lmat: "left matrix" of size len(smile) X num_elements
        - cmat: "connectivity matrix" of size num_elements X len(smile)
    '''
    try:
        indices, symbols, cmats = zip(*get_full_neighbor_vectors(smile))
    except:
        ### Need to write fallback method when RDKit fails
        print(f"Skipping smile {smile} due to RDKit error.")
        return None, None
    
    lmat = np.vstack([get_lmat_vec(symbol, num_elements) for symbol in symbols])
    cmat = np.hstack(cmats)
    return lmat, cmat

def process_nodes(node_raw_data, vectors):
    node_data = []
    temp_list = []
    
    for n in node_raw_data:
        if n['atom_type'] == 'H':
            continue
        temp_list += [n]
    node_data_no_H = tuple(temp_list)

    assert len(vectors) == len(node_data_no_H), f'{len(vectors)} != {len(node_data_no_H)}'
    for node_idx, data in enumerate(zip(vectors, node_data_no_H)):
        v, n = data
        idx, symbol, cmat = v
         
        lmat = np.zeros((1,EN_MAT.shape[0]))
        lmat[0, Chem.Atom(symbol).GetAtomicNum() - 1] = 1
        
        e_neg_score = np.einsum('ij,jk,ki->i', lmat, EN_MAT, cmat)[0]        
        
        symbol = n['atom_type']
        formal_charge = n['formal_charge']
        
        for orb, binding_e in zip(n['orbitals'], n['binding_energies']):
            if orb == -1 or binding_e == -1:
                continue
            ref_e = -giveorbitalenergy(symbol, orb)
            n, l = get_n_l(orb)
            # Binding energy normalized against reference energy
            # AKA, train against the difference from reference atomic orbital energy
            # Each entry is as follows:
            # [Atomic Num, formal charge, e_neg_score, quantum number 'n', quantum number 'l', binding_e - atomic_orbital_e]
            # one-hot encoding of atomic number
            temp_arr = lmat.flatten().tolist()
            temp_arr += [formal_charge, e_neg_score, n, l, binding_e-ref_e]
            node_data += [np.array(temp_arr,dtype=np.float32)]
    node_data = np.atleast_2d(node_data)   
    return node_data

def networkx2arr(data):
    smiles = list(data.keys())
    
    # For each networkx graph, process the nodes and edges into torch_geometric.data.Data graphs
    graph_data = []
    for graph_num, smile in enumerate(smiles):
        graph = data[smile]
        print(f"Processing graph {graph_num+1}/{len(smiles)}: {smile}")
        
        try:
            vectors = get_full_neighbor_vectors(smile)
        except:
            print(f"Skipping smile {smile} due to RDKit error")
            continue
        nodes, node_raw_data = zip(*graph.nodes(data=True))
        # Hydrogen will never have a core binding energy
        # For atom-centric data formatting, Hydrogen may be safely omitted
        temp_list = []
        for n in node_raw_data:
            if n['atom_type'] == 'H':
                continue
            temp_list += [n]
        node_raw_data = tuple(temp_list)
        
        ### Process Node information
        new_data = process_nodes(node_raw_data, vectors)
        if new_data.size != 0:
            graph_data += [np.atleast_2d(new_data)]
        else:
            print("Found no information in this graph")
    return np.vstack(graph_data)

class NeuralNetwork(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch * batch_size >= size - batch_size:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred.flatten(), y.flatten()).item()

    test_loss /= num_batches
    print(f"Test Avg loss: {test_loss:>8f} \n")
    return test_loss

if __name__=='__main__':
    # Input arguments
    in_filename = sys.argv[1] # networkx .json file
    
    learning_rate = 1e-4
    batch_size = 16
    epochs = 50
    loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()

    # Load in networkx graphs
    data = load_data_from_file(in_filename)
    graph_data = networkx2arr(data)
    unique_data = np.unique(graph_data, axis=0)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using {device} device")
    model = NeuralNetwork(in_dim=graph_data[0].shape[0]-1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    

    # np.random.seed()
    indices = np.arange(unique_data.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:len(indices)*9//10]
    test_idx = indices[len(indices)*9//10:]

    train_data_tensor = torch.tensor(unique_data[train_idx,:-1], dtype=torch.float32).to(device)
    train_labels_tensor = torch.tensor(unique_data[train_idx,-1], dtype=torch.float32).to(device)
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data_tensor = torch.tensor(unique_data[test_idx,:-1], dtype=torch.float32).to(device)
    test_labels_tensor = torch.tensor(unique_data[test_idx,-1], dtype=torch.float32).to(device)
    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    temp = []
    for X, y in test_dataloader:
        pred = model(X).flatten()
        for n in (pred-y).cpu().detach().numpy(): 
            temp += [n]
    
    stats = {
    'MAE': np.mean(np.abs(temp)),
    'STDEV': np.std(temp),
    'RMSE': np.sqrt(np.mean(np.array(temp)**2)),
    'MSE': np.mean(temp), # Mean-Signed Error
    'Max Error': np.max(temp)}

    print("\nError Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
