import pandas as pd
import numpy as np

try: 
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit-pypi"])
    # subprocess.check_call([sys.executable, "-m", "conda", "install", "rdkit", "-c conda-forge"])
    
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys


def smiles2fing(smiles):
    ms_tmp = [Chem.MolFromSmiles(i) for i in smiles]
    ms_none_idx = [i for i in range(len(ms_tmp)) if ms_tmp[i] == None]
    
    ms = list(filter(None, ms_tmp))
    
    maccs = [MACCSkeys.GenMACCSKeys(i) for i in ms]
    maccs_bit = [i.ToBitString() for i in maccs]
    
    fingerprints = pd.DataFrame({'maccs': maccs_bit})
    fingerprints = fingerprints['maccs'].str.split(pat = '', n = 167, expand = True)
    fingerprints.drop(fingerprints.columns[0], axis = 1, inplace = True)
    
    colname = ['maccs_' + str(i) for i in range(1, 168)]
    fingerprints.columns = colname
    fingerprints = fingerprints.astype(int).reset_index(drop = True)
    
    return ms_none_idx, fingerprints

if __name__ == '__main__':
    folder = './Data'
    file = 'tox21.xlsm'
    cur_sheet = 'Tox21'
    data = pd.read_excel(f'{folder}/{file}', sheet_name= cur_sheet)
    print(data.head())
    smiles = data['smiles'].to_numpy()
    _, fings = smiles2fing(smiles)
    mol_id = data['mol_id']
    labels = data.iloc[:,0:12]
    dataset = pd.concat([mol_id, fings, labels], axis= 1)
    dataset.to_csv(f'{folder}/Tox21_Dataset.csv', index= False)

