
from rdkit import Chem
import numpy as np

def get_full_neighbor_vectors(smiles, num_elements=100, add_bonds=True):
    '''
    This function will generate a vector of size (100, 1) 
    for all non-H atoms in given smile notation. Where each index 
    number denotes the number of that atoms in the surrounding of 
    the current atom.
    For example: CH4:
    C: [4, 0, 0, ...0]
    For CO
    C: [0, 0, 0, 0, 0, 0, 0, 1, 0, ...0]
    O: [0, 0, 0, 0, 0, 1, 0, 0, ....0]

    If add_bonds is turned on, it multiplies the bonds with neighbors
    For example: Propene (H2C=CH-CH3)
    C:[2,0,0,0,0,2,....] first carbon
    C:[1,0,0,0,0,3,....] second carbon
    C:[3,0,0,0,0,1,....] third carbon
    '''
    try: 
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
    except:
        mol = Chem.MolFromSmiles(smiles,sanitize=False) ## for bad cases, doesn't always get the right answer

    atom_vectors = []

    for atom in mol.GetAtoms():

        if atom.GetSymbol() == 'H':
            continue 

        vec = np.zeros((num_elements, 1), dtype=int)

        for nbr in atom.GetNeighbors():
            atomic_num = nbr.GetAtomicNum()
            if add_bonds==True:
                bond_order = mol.GetBondBetweenAtoms(atom.GetIdx(),nbr.GetIdx()).GetBondTypeAsDouble()
            else: 
                bond_order = 1
            if atomic_num < num_elements:
                vec[atomic_num-1][0] += 1*bond_order

        atom_vectors.append((atom.GetIdx(), atom.GetSymbol(), vec))

    return atom_vectors


def get_second_neighbors_vector(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    atom_vectors = []

    for atom in mol.GetAtoms():

        if atom.GetSymbol() == 'H':
            continue 

        vec = np.zeros((100, 1), dtype=int)
        second_nn = {}
        for nbr in atom.GetNeighbors():
            atomic_num = nbr.GetAtomicNum()
            if atomic_num < 100:
                vec[atomic_num-1][0] += 1
            
            newvec = np.zeros((100, 1), dtype=int)
            for nbr_ in nbr.GetNeighbors():
                atomic_num = nbr.GetAtomicNum()
                if atomic_num < 100:
                    newvec[atomic_num-1][0] += 1
            if (atomic_num-1) not in second_nn:
                second_nn[atomic_num-1] = newvec
            else:   
                second_nn[atomic_num-1] += newvec

        atom_vectors.append((atom.GetIdx(), atom.GetSymbol(), vec, second_nn))
   
    return atom_vectors


if __name__=='__main__':
    smiles = "CCO" # #CH3CH2OH
    vectors = get_second_neighbors_vector(smiles)

    for idx, symbol, first_nn, second_nn in vectors:
        print(f"Atom {idx} ({symbol})")
        print(first_nn.T)

        for k in second_nn.keys():
            print(second_nn[k].shape)

