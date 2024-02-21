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
import matplotlib.pyplot as plt


# 데이터 파일 경로 설정
folder = '.'
file = 'tox21_dataset.csv'
path = f'{folder}/{file}'

# 데이터 파일 불러오기
df = pd.read_csv(path)

# 전체 데이터 수 파악
print('전체 데이터 갯수:', len(df.index))

index_label = 'NR-AR'    # NR_AR label의 index

# 각 label에 해당하는 데이터의 수 구하기
    # loc:  인덱스(index)나 컬럼(column)의 이름을 기준으로 행과 열을 선택합니다.
    # iloc: 위치(인덱스나 컬럼의 숫자)를 기준으로 행과 열을 선택합니다.
label_0 = len(df.loc[df.loc[:,'NR-AR'] == 0])  # label이 0인 데이터
label_1 = len(df.loc[df.loc[:,'NR-AR'] == 1])  # label이 1인 데이터
label_nan = len(df[df.loc[:, index_label].isna()])  # label이 결측값인 데이터


print('NR-AR의 각 class별 갯수')
print('  - 0인 데이터의 수:',label_0)
print('  - 1인 데이터의 수:',label_1)
print('  - 결측값인 데이터의 수:',label_nan)
print()

# bar 그래프를 통해 data imbalance를 시각화
plt.bar([0,1,2], [label_0, label_1, label_nan])
plt.xticks([0,1,2], [0,1,'NaN'])
plt.title(f'{index_label}')
plt.xlabel('label')
plt.ylabel('Num of Data')
plt.show()

index_label = 43    # 2(maccs_1) ~ 167(maccs_166) 중 선택

# 각 label에 해당하는 데이터의 수 구하기
label_0 = len(df.loc[df.iloc[:,index_label] == 0])
label_1 = len(df.loc[df.iloc[:,index_label] == 1])

print(f'maccs_{index_label}의 분포')
print('  - 0인 데이터의 수:',label_0)
print('  - 1인 데이터의 수:',label_1)
print()

# bar 그래프를 통해 data imbalance를 시각화
plt.bar([0,1], [label_0, label_1])
plt.xticks([0,1], [0,1])
plt.title(f'{df.columns[index_label]}')
plt.xlabel('label')
plt.ylabel('Num of Data')
plt.show()

# 데이터 train/test 분리하기
# 데이터 분리를 하기 위한 하이퍼 파라미터 설정
label_idx = 0       # 학습에 사용하고 하는 label 선택 - 0 ~ 11중 선택 가능 | 0 -> NR-AR, 11 -> SR-p53
train_size = 0.8    # train과 test의 비율을 설정


feature = df.iloc[:,2:168]          # Features (166개)
label = df.iloc[:,168+label_idx]    # label

# label 기준으로 결측값(NaN)인 데이터 제거하기
idx_none = df[df.iloc[:,168 + label_idx].notna()].index # 결측값이 아닌 데이터의 index
feature = df.iloc[idx_none, 2:168]          # 필요한 데이터의 feature만 가져오기
label = df.iloc[idx_none, 168+label_idx]    # 필요한 데이터의 label만 가져오기


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
