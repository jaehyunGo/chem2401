# 라이브러리 호출
import pandas as pd
import numpy as np

# rdkit: 화학 정보학과 기계 학습을 위한 오픈 소스 화학 정보학 소프트웨어 툴킷입니다.

try:
# rdkit.Chem: 분자와 관련된 기본적인 기능과 클래스를 제공합니다.
# rdkit.Chem.MACCSkys: 화합물의 분자 지문을 생성하는 데 사용되는 방법 중 하나입니다.
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys

# rdkit 라이브러리가 호출되지 않을 시, 라이브러리 설치 후 재호출
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

    colname = ['maccs_' + str(i) for i in range(167)]
    fingerprints.columns = colname
    fingerprints = fingerprints.astype(int).reset_index(drop = True)

    return ms_none_idx, fingerprints

# 데이터를 불러오기 위한 경로 설정
folder = '.'
file = 'tox21.xlsm' # 12개의 label과 SMILE가 존재하는 데이터 파일
cur_sheet = 'Tox21'

# pandas 라이브러리를 통해 csv파일을 불러오기
data = pd.read_excel(f'{folder}/{file}', sheet_name= cur_sheet)
print()
print('='*100)
print('불러온 데이터파일(tox21.xlsm) 데이터 앞부분 확인')
print(data.head())  # 데이터 일부분을 출력
# 가공하고자 하는 데이터를 따로 추출
smiles = data['smiles'].to_numpy()
# smiles2fing 함수를 통해 데이터 변환 실시
_, fings = smiles2fing(smiles)  # fings 변수에 변환된 features가 저장되어 있음

# 학습에 사용할 수 있도록 데이터셋 생성
mol_id = data['mol_id']     # 이름
labels = data.iloc[:,0:12]  # label
dataset = pd.concat([mol_id, fings, labels], axis= 1)   # features

# 생성한 데이터셋을 csv파일로 내보내기
dataset.to_csv(f'{folder}/tox21_dataset.csv', index= False)

print()
print('='*100)
print('\n 생성한 dataset 앞부분')
print(dataset.head())

# train/test 나누기

from sklearn.model_selection import train_test_split


# 데이터 파일 경로 설정
folder = '.'
file = 'tox21_dataset.csv'
path = f'{folder}/{file}'

# 데이터 분리를 하기 위한 하이퍼 파라미터 설정
label_idx = 0       # 학습에 사용하고 하는 label 선택 - 0 ~ 11중 선택 가능 | 0 -> NR-AR, 11 -> SR-p53
train_size = 0.8    # train과 test의 비율을 설정

# 데이터 파일 불러오기
df = pd.read_csv(path)
feature = df.iloc[:,1:167]          # Features (166개)
label = df.iloc[:,167+label_idx]    # label

# label 기준으로 결측값(NaN)인 데이터 제거하기
idx_none = df[df.iloc[:,167 + label_idx].notna()].index # 결측값이 아닌 데이터의 index
feature = df.iloc[idx_none, 1:167]          # 필요한 데이터의 feature만 가져오기
label = df.iloc[idx_none, 167+label_idx]    # 필요한 데이터의 label만 가져오기


# 사이킷런에서 제공하는 함수를 이용하여 train과 test 데이터 분리하기
    # stratify : label을 기준으로 train과 test의 분포가 동일하도록 분리함
train_x, test_x, train_y, test_y = train_test_split(feature, label, train_size= train_size, stratify= label, random_state= 42)

print('='*100)
print('Oversampling전 데이터 shape 확인')
print('train dataset')
print(f'\tX : {train_x.shape}, Y : {train_y.shape}')
print('test dataset')
print(f'\tX : {test_x.shape}, Y : {test_y.shape}')


# Oversampling
    # SMOTE 방식을 통해 데이터 불균형 해결
    # SMOTE(Synthetic Minority Over-sampling Technique)는 불균형한 클래스 분포를 가진 데이터셋에서 소수 클래스를 오버샘플링하기 위한 기법으로, 소수 클래스의 샘플을 합성하여 데이터를 증가시키는 방법입니다.
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state= 42)
X_resampled, y_resampled = smote.fit_resample(train_x, train_y) # train 데이터를 늘려주기

train_x = X_resampled
train_y = y_resampled
# 분리한 결과 확인하기
print()
print('='*100)
print('Oversampling후 데이터 shape 확인')
print('train dataset')

# train 데이터 파일 생성하기
print(f'\tX : {train_x.shape}, Y : {train_y.shape}')
train_x = pd.DataFrame(train_x)
train_y = pd.DataFrame(train_y)
dataset_train = pd.concat([train_x, train_y], axis= 1)
dataset_train.to_csv(f'{folder}/tox21_train.csv', index= False)

# test 데이터 파일 생성하기
print('test dataset')
print(f'\tX : {test_x.shape}, Y : {test_y.shape}')
test_x = pd.DataFrame(test_x)
test_y = pd.DataFrame(test_y)
dataset_test = pd.concat([test_x, test_y], axis= 1)
dataset_test.to_csv(f'{folder}/tox21_test.csv', index= False)
