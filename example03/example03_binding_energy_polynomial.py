import sys
import os
import json
import networkx as nx
import numpy as np
from smiletovectors import get_full_neighbor_vectors
from paulingelectro import get_eleneg_diff_mat
from scipy import optimize
from rdkit import Chem

'''
This file provides functionality for fitting to the function:
y = mx + b

y := experimental binding energy - atomic orbital energy
m := weights to train (1 parameter per element)
x := electronegativity environment 
b := Constant (1 parameter per element)
'''

au2eV = 27.21139

### Helper functions to read data
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
		graph = nx.node_link_graph(string_dict[key], link="edges")
		result_dict[key] = graph
	return result_dict

### Helper functions to get quantum numbers of each orbital
def get_l(l):
    '''
    l = s, p, d, f, g
    '''
    return {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g':4,}.get(l, "l value is not valid")

def get_n_l(orb):
    n = int(orb[0])
    l = get_l(str(orb[1]))
    return n, l

### For a given element and orbital, return the atomic orbital energy in eV
def giveorbitalenergy(ele, orb, orbital_energy_file='orbitalenergy.json'):
    with open(orbital_energy_file, 'r') as f:
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

def process_nodes(data, smile, en_mat, num_elements=100):
    '''
    Provided the raw data for a node into a matrix form with our desired features/labels
    Arguments:
        - data: networkx graph
        - smile: SMILE string to use as key in 'data'
        - en_mat: electronegativity difference matrix
    returns:
        matrix of size n_samples X n_features+n_labels
    '''
    graph = data[smile]    
    try:
        vectors = get_full_neighbor_vectors(smile, num_elements=num_elements)
    except:
        print(f"Skipping smile {smile} due to RDKit error")
        return None
    
    nodes, node_raw_data = zip(*graph.nodes(data=True))
    node_data = []
    
    # Hydrogen will never have a core binding energy
    # For atom-centric data formatting, Hydrogen may be safely omitted
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
        
        lmat = np.zeros((1,en_mat.shape[0]))
        lmat[0, Chem.Atom(symbol).GetAtomicNum() - 1] = 1
        
        e_neg_score = np.einsum('ij,jk,ki->i', lmat, en_mat, cmat)[0]
        
        atom_type = n['atom_type']
        formal_charge = n['formal_charge']
        
        for orb, binding_e in zip(n['orbitals'], n['binding_energies']):
            if orb == -1 or binding_e == -1:
                continue
            ref_e = -giveorbitalenergy(symbol, orb)
            n, l = get_n_l(orb)
            assert symbol == atom_type
            # Binding energy zeroed against reference energy
            # AKA, train against the difference from reference atomic orbital energy
            # Each entry is as follows:
            # [Atomic Num, formal charge, e_neg_score, quantum number 'n', quantum number 'l', atomic_orbital_e, binding_e]
            node_data += [np.array([Chem.Atom(symbol).GetAtomicNum(), formal_charge, e_neg_score, n, l, ref_e, binding_e],dtype=np.float64)]
    return node_data

def networkx2arr(data, num_elements=100):
    smiles = list(data.keys())
    en_mat = get_eleneg_diff_mat(num_elements)

    graph_data = []
    for graph_num, smile in enumerate(smiles):
        print(f"Processing graph {graph_num+1}/{len(smiles)}: {smile}")
        ### Process Node information
        new_data = process_nodes(data, smile, en_mat, num_elements=num_elements)
        if new_data is None:
            continue
        
        if len(new_data) != 0:
            graph_data += [np.atleast_2d(new_data)]
        else:
            print(f"Found no information in {smile}")

    return np.vstack(graph_data)

def get_xvec(weights, element_list):
    return np.array([weights[int(el)-1] for el in element_list])

def test_model_p2(weights, elements, e_neg, ref_e, exp_e):
    '''
    Return the testing loss for Polynomial Model 2 with weights 'weights'
        - model_p2: Fit the function y = mx + b
        - y = exp_e - ref_e
        - m = weights
        - x = electronegativity environment 'e_neg'
        - b = constant for a given element
    Arguments:
        - weights: 1D array of length 'num_element'*2
        - elements: 1D array of atomic numbers
        - e_neg: 1D array of electronegativity environments
        - ref_e: Atomic orbital energy (reference)
        - exp_e: Experimental binding energy
    Returns:
        predict: Predictions
        loss: RMSE Error
    '''        
    if len(elements) == 0: 
        return np.array([]), np.array([])
    xvec = get_xvec(weights.reshape(-1,2), elements)
    predict = np.sum(np.vstack((e_neg, np.ones(e_neg.shape))).T * xvec,axis=1) + ref_e
    # loss = np.sqrt(np.mean((predict - exp_e) ** 2))
    error = predict - exp_e
    return predict, error


def train_model_p2(elements, e_neg, ref_e, exp_e, num_elements=100):
    '''
    Return the optimized weights for fitting Polynomial Model 1
        - model_p2: Fit the function y = mx + b
        - y = exp_e - ref_e
        - m = weights
        - x = electronegativity environment 'e_neg'
        - b = constant for a given element
    Arguments:
        - elements: 1D array of atomic numbers
        - e_neg: 1D array of electronegativity environments
        - ref_e: Atomic orbital energy (reference)
        - exp_e: Experimental binding energy
        - num_elements: Number of elements to compute weights for (fields will be 0 for elements not encountered in training)
    Returns:
        weights: 1D array of length 'num_element'
    '''
    weights = np.zeros(num_elements*2)
    def errorfunc(x):
        xvec = get_xvec(x.reshape(-1,2), elements)
        arr = np.vstack([e_neg, np.ones(e_neg.shape)]).T
        loss = np.sqrt(np.mean((np.sum(arr * xvec, axis=1) + ref_e - exp_e) ** 2))
        return loss
    results = optimize.minimize(errorfunc, weights)
    weights = results['x']
    loss = results['fun']
    return weights, loss

if __name__=='__main__':
    # Input arguments
    in_filename = 'graph_data.json'
    
    # Load in networkx graphs
    data = load_data_from_file(in_filename)
    num_elements = 20
    # [Atomic Num, formal charge, e_neg_score, quantum number 'n', quantum number 'l', atomic_orbital_e, binding_e]
    graph_data = networkx2arr(data, num_elements)
    
    null_loss = np.sqrt(np.mean((graph_data[:,5] - graph_data[:,6])**2))
    print(f"Null Loss: {null_loss:.3f}eV")

    train_samples = 700
    test_samples = graph_data.shape[0] - train_samples
    
    print(f"Total number of binding energies: {graph_data.shape[0]}")

    ### Training
    exp_energies = graph_data[:train_samples,6]
    ref_energies = graph_data[:train_samples, 5]
    lemcmat = graph_data[:train_samples, 2]
    element_list = graph_data[:train_samples, 0].flatten()
    
    weights, loss = train_model_p2(element_list, lemcmat, ref_energies, exp_energies)
    print(f"Training RMSE over {train_samples} samples: {loss:.3f}eV")
    
    ### Testing
    exp_energies = graph_data[train_samples:train_samples+test_samples, 6]
    ref_energies = graph_data[train_samples:train_samples+test_samples, 5]
    exp_minus_ref = exp_energies - ref_energies
    element_list = graph_data[train_samples:train_samples+test_samples, 0]
    lemcmat = graph_data[train_samples:train_samples+test_samples, 2]

    predict, errors = test_model_p2(weights, element_list, lemcmat, ref_energies, exp_energies)
    predict_loss = np.sqrt(np.mean((errors) ** 2))
    print(f"Testing RMSE over {test_samples} samples: {predict_loss:.3f}eV")

    # Error statisitcs
    mae = np.mean(np.abs(errors))
    stdev = np.std(errors)
    rmse = np.sqrt(np.mean(errors**2))
    mean_error = np.mean(errors)
    max_error = np.max(np.abs(errors))

    stats = {
        'MAE': mae,
        'STDEV': stdev,
        'RMSE': rmse,
        'MSE': mean_error, # Mean-Signed Error
        'Max Error': max_error}

    print("\nError Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    
    # # Plotting the erros
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.bar(stats.keys(), stats.values())
    # ax.set_ylabel('Error Value')
    # ax.set_title('Error Statistics')
    # plt.xticks(rotation=45)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.show()



